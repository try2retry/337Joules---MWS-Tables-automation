import re
import json
import time
import socket
import logging
import ipaddress
import threading
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin

import requests
import pandas as pd

import ai_client

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------------------------------------------------------

# Курсы валют
DEFAULT_RATES_URL = "https://www.cbr-xml-daily.ru/daily_json.js"
# Если пользователь дал только корень сайта (https://cbr.ru), дописываем известный адрес с курсами
KNOWN_RATE_ENDPOINTS = {"cbr-xml-daily.ru": "/daily_json.js", "cbr.ru": "/scripts/XML_daily.asp"}
RATES_TTL = 3600  # секунд хранить курсы в памяти процесса
FAILED_TTL = 300  # после неудачной загрузки не долбим недоступный адрес (и не ждем таймаут на каждой таблице)
REQUEST_TIMEOUT = 3
MAX_RESPONSE_BYTES = 1_000_000  # курсы весят десятки КБ; больше - подозрительно
MAX_REDIRECTS = 3
MAX_CACHE_ENTRIES = 16  # адреса задает пользователь - ограничиваем размер кэша
FALLBACK_RATES = {"RUB": 1.0, "USD": 90.0, "EUR": 98.0, "CNY": 12.5}  # рублей за 1 единицу валюты
DEFAULT_CURRENCY = "RUB"

# Подсказки для определения валюты по заголовку колонки / значению (порядок важен: длинные признаки раньше)
CURRENCY_HINTS = [("usd", "USD"), ("доллар", "USD"), ("$", "USD"), ("eur", "EUR"), ("евро", "EUR"), ("€", "EUR"),
                  ("cny", "CNY"), ("юан", "CNY"), ("yuan", "CNY"), ("rmb", "CNY"), ("¥", "CNY"),
                  ("rub", "RUB"), ("руб", "RUB"), ("₽", "RUB")]

# Единицы измерения: слово -> (группа, коэффициент к базовой единице группы: кг, м, л, шт)
UNITS = {
    "г": ("mass", 0.001), "гр": ("mass", 0.001), "грамм": ("mass", 0.001), "g": ("mass", 0.001),
    "кг": ("mass", 1.0), "kg": ("mass", 1.0), "т": ("mass", 1000.0), "тн": ("mass", 1000.0),
    "мм": ("length", 0.001), "mm": ("length", 0.001), "см": ("length", 0.01), "cm": ("length", 0.01),
    "м": ("length", 1.0), "m": ("length", 1.0), "км": ("length", 1000.0), "km": ("length", 1000.0),
    "мл": ("volume", 0.001), "ml": ("volume", 0.001), "л": ("volume", 1.0), "l": ("volume", 1.0),
    "шт": ("pieces", 1.0), "штук": ("pieces", 1.0), "штука": ("pieces", 1.0), "pcs": ("pieces", 1.0)
}
UNITS.update({
    "мг": ("mass", 1e-6), "килограмм": ("mass", 1.0), "тонна": ("mass", 1000.0),
    "дм": ("length", 0.1), "метр": ("length", 1.0), "сантиметр": ("length", 0.01), "миллиметр": ("length", 0.001),
    "литр": ("volume", 1.0), "миллилитр": ("volume", 0.001),
    "пара": ("pieces", 2.0), "дюжина": ("pieces", 12.0), "дюж": ("pieces", 12.0)
})
DEFAULT_TARGETS = {"mass": "кг", "length": "м", "volume": "л", "pieces": "шт"}

# Множители-слова: "тыс. шт" = 1000 шт, "5 тыс. шт" = 5000 шт, "2 млн" = 2 000 000
SCALE_RE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*)?\b(тыс|млн|миллион)[а-я]*\.?")
SCALES = {"тыс": 1e3, "млн": 1e6, "миллион": 1e6}
# Умножение количеств: "2*500 г", "2 x 500", "2х500", "2 шт × 500 г" (латинская x, кириллическая х, ×, *)
TIMES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:шт|штук|уп[а-я]*)?\s*[*x×х]\s*(\d+(?:\.\d+)?)")

# Пользовательские правила перевода в конфигураторе: "бухта = 100 м; ящик = 12 шт; уп = 10 шт"
RULE_RE = re.compile(r"^\s*(?:\d+(?:[.,]\d+)?\s*)?([а-яa-z.]+)\s*=\s*(\d+(?:[.,]\d+)?)?\s*([а-яa-z]+)\s*$")
CUSTOM_RULES_HEADER_PARTS = ("свои", "пользоват", "правил", "пересч", "конверт")

# Сколько уникальных строк за один прогон отправляем ИИ, и по сколько в одном запросе (стоимость и задержка)
AI_MAX_ITEMS = 200
AI_BATCH_SIZE = 50
AI_AMOUNT_RANGE = (1e-6, 1e7)  # разумные границы ответа ИИ (количество в единицах)
_ai_cache = {}  # {ключ строки: (группа, база) | None}: чтобы не спрашивать ИИ про одно и то же в каждой таблице
MAX_AI_CACHE = 5000

# Названия колонок конфигуратора (после нормализации заголовка) -> группа единиц
CONFIG_UNIT_PARTS = {"масс": "mass", "длин": "length", "объем": "volume", "шту": "pieces"}

# Упаковки: слово в колонке unit, а количество внутри берем из самой единицы ("уп 10 шт") или из названия товара
PACK_PREFIXES = ("уп", "пач", "кор", "блок", "набор", "компл")
UNIT_PART_RE = re.compile(r"(\d+(?:\.\d+)?)?\s*([а-яa-z]+)")  # "10 шт", "100 г", "кг"
# Название товара: слово-упаковка + (по) + число + единица ("уп. по 50 шт", "пачка 500 г")
NAME_PACK_RE = re.compile(r"(?:уп|пач|кор|блок|набор|фас)[а-я]*\.?\s*(?:по\s*)?[xх*]?\s*(\d+(?:[.,]\d+)?)\s*([а-яa-z]+)")
# Запасной вариант: просто "50 шт" / "500 г" в названии (длину не берем - это размер товара, а не содержимое)
NAME_AMOUNT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(шт|штук|гр?|кг|мл|л)\b")

# Колонки конфигуратора, в которых может лежать ссылка на источник курсов
RATES_URL_HEADER_PART = "url"

_cache = {}  # {url: (время_истечения, {валюта: курс})}
_cache_lock = threading.Lock()


# ----------------------------------------------------------------------------------------------------------------------
# Валюты и курсы
# ----------------------------------------------------------------------------------------------------------------------

# Код валюты, если в тексте есть ее признак ("Цена, $", "12 руб", "€5"), иначе пустая строка
def find_currency(text) -> str:
    text = str(text or "").strip().lower()

    for hint, code in CURRENCY_HINTS:
        if hint in text:
            return code

    return ""


# Код валюты по заголовку колонки или значению ячейки. Признака нет - рубли
def detect_currency(text) -> str:
    return find_currency(text) or DEFAULT_CURRENCY


# Полный адрес источника курсов: корень сайта ЦБ дополняем нужным путем
def resolve_rates_url(url: str) -> str:
    parsed = urlparse(url.strip())

    if parsed.path in ("", "/"):
        for host, path in KNOWN_RATE_ENDPOINTS.items():
            if (parsed.hostname or "").endswith(host):
                return parsed._replace(path=path).geturl()

    return url.strip()


# Защита от SSRF: адрес задает пользователь, а запрос идет с нашего сервера.
# Разрешаем только https и только публичные IP (никаких localhost, 10.x, 169.254.x и т.п.)
def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False

    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except OSError:
        return False

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast \
                or ip.is_unspecified:
            return False

    return True


# Разбирает ответ с курсами: JSON (cbr-xml-daily.ru) или XML ЦБ РФ. Возвращает {код: рублей за 1 единицу}
def parse_rates(content: bytes) -> dict:
    rates = {}

    if content.lstrip().startswith(b"{"):
        for code, item in json.loads(content)["Valute"].items():
            rates[code] = float(item["Value"]) / int(item.get("Nominal", 1))
    else:
        # DTD и внешние сущности в курсах не нужны: отсекаем их до разбора (защита от XML-бомб)
        if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
            raise ValueError("DTD в XML запрещен")

        for valute in ET.fromstring(content).iter("Valute"):
            value = float(valute.findtext("Value").replace(",", "."))
            rates[valute.findtext("CharCode")] = value / int(valute.findtext("Nominal") or 1)

    if not rates:
        raise ValueError("В ответе нет курсов")

    return {**rates, "RUB": 1.0}


# Скачивает курсы по адресу: таймаут 3 с, ограничение размера. Редиректы (cbr-xml-daily.ru -> www.) идем вручную,
# чтобы каждый следующий адрес тоже прошел проверку на SSRF
def fetch_rates(url: str) -> dict:
    for _ in range(MAX_REDIRECTS + 1):
        if not is_safe_url(url):
            raise ValueError("Небезопасный адрес источника курсов")

        with requests.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=False, stream=True) as resp:
            if resp.is_redirect:
                url = urljoin(url, resp.headers.get("Location", ""))
                continue

            resp.raise_for_status()
            content = resp.raw.read(MAX_RESPONSE_BYTES + 1, decode_content=True)

        if len(content) > MAX_RESPONSE_BYTES:
            raise ValueError("Слишком большой ответ")

        return parse_rates(content)

    raise ValueError("Слишком много редиректов")


# Курсы с кэшем в памяти процесса (TTL 1 час): при обработке 10 таблиц - один сетевой запрос.
# Любой сбой -> резервные курсы + предупреждение, код не падает
def get_rates(url: str = None) -> dict:
    url = resolve_rates_url(url or DEFAULT_RATES_URL)

    # Лок держим и на время запроса: параллельные запросы дождутся первого, а не пойдут в сеть все разом
    with _cache_lock:
        cached = _cache.get(url)
        if cached and cached[0] > time.time():
            return cached[1]

        try:
            rates = {**FALLBACK_RATES, **fetch_rates(url)}  # чего нет в источнике - берем из резерва
            ttl = RATES_TTL
        except (requests.RequestException, ValueError, KeyError, TypeError, ET.ParseError) as e:
            log.warning("Курсы валют по %s недоступны (%s), используем резервные", urlparse(url).netloc,
                        type(e).__name__)
            rates, ttl = dict(FALLBACK_RATES), FAILED_TTL

        if len(_cache) >= MAX_CACHE_ENTRIES:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]))  # вытесняем самую старую запись

        _cache[url] = (time.time() + ttl, rates)
        return rates


# ----------------------------------------------------------------------------------------------------------------------
# Конфигуратор из MWS Tables
# ----------------------------------------------------------------------------------------------------------------------

# Первое непустое значение колонки
def first_value(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    return ""


# Пары (параметр, значение) из конфигуратора. Поддерживаем два вида таблицы:
#  1) колонками: заголовок = параметр, под ним значения ("Валюта" -> USD; "Свои_единицы" -> несколько правил);
#  2) списком: колонки "Параметр" и "Значение", каждая строка - один параметр
def config_pairs(config_df: pd.DataFrame) -> list:
    keys = [c for c in config_df.columns if re.search(r"парам|ключ|настрой|key", str(c).lower())]
    values = [c for c in config_df.columns if re.search(r"знач|value", str(c).lower())]

    if keys and values:
        rows = zip(config_df[keys[0]], config_df[values[0]])
        return [(str(k), str(v).strip()) for k, v in rows if pd.notna(k) and pd.notna(v) and str(v).strip()]

    return [(str(h), str(v).strip()) for h in config_df.columns for v in config_df[h] if pd.notna(v) and str(v).strip()]


# Добавляет пользовательские правила вида "бухта = 100 м; ящик = 12 шт; 1 уп = 10 шт" (через ; или перенос строки)
def add_custom_rules(config: dict, text: str):
    for part in re.split(r"[;\n]", text):
        if not part.strip():
            continue

        match = RULE_RE.match(expand_amounts(part))  # заодно понимает "= 1 тыс шт" и "= 2*500 г"
        if not match or match.group(3) not in UNITS:
            log.warning("Не удалось разобрать правило перевода единиц: '%s'", part.strip())
            continue

        group, factor = UNITS[match.group(3)]
        config["custom_units"][match.group(1).strip(".")] = (group, float(match.group(2) or 1) * factor)


# Настройки пользователя из config_df: целевая валюта, целевые единицы, свои правила перевода, ссылка на курсы.
# Все поля необязательны; для одиночных настроек берется первое значение
def parse_config(config_df) -> dict:
    config = {"currency": DEFAULT_CURRENCY, "targets": dict(DEFAULT_TARGETS), "rates_url": None, "custom_units": {}}
    if config_df is None or config_df.empty:
        return config

    done = set()
    for header, value in config_pairs(config_df):
        key = re.sub(r"[\s_\-()]", "", header.lower()).replace("ё", "е")

        if any(part in key for part in CUSTOM_RULES_HEADER_PARTS):
            add_custom_rules(config, value)
            continue

        group = next((g for part, g in CONFIG_UNIT_PARTS.items() if part in key), None)
        kind = "currency" if "валют" in key else group or ("url" if RATES_URL_HEADER_PART in key else None)
        if kind is None or kind in done:
            continue

        done.add(kind)
        if kind == "currency":
            config["currency"] = value.upper() if re.fullmatch(r"[A-Za-z]{3}", value) else detect_currency(value)
        elif kind == "url":
            config["rates_url"] = value
        else:
            unit = value.lower().strip(". ")

            if UNITS.get(unit, (None,))[0] == kind:
                config["targets"][kind] = unit
            else:
                log.warning("Единица '%s' не подходит для группы '%s', оставляем '%s'", value, kind,
                            config["targets"][kind])

    return config



# ----------------------------------------------------------------------------------------------------------------------
# Единицы измерения и упаковки
# ----------------------------------------------------------------------------------------------------------------------

# Приводит текст с количеством к простому виду: нижний регистр, "," -> ".", "5 тыс. шт" -> "5000 шт", "2*500 г" -> "1000 г"
def expand_amounts(text) -> str:
    text = str(text or "").lower().replace(",", ".").replace("ё", "е")
    text = SCALE_RE.sub(lambda m: f"{float(m.group(1) or 1) * SCALES[m.group(2)]:.10g} ", text)

    # Цепочки "2*3*500" сворачиваем по одному умножению за проход
    previous = None
    while previous != text:
        previous = text
        text = TIMES_RE.sub(lambda m: f"{float(m.group(1)) * float(m.group(2)):.10g}", text, count=1)

    return text


# Разбирает содержимое колонки unit. Возвращает (группа, количество в базовых единицах группы за ОДНУ продаваемую
# единицу, это_упаковка, взято_из_правила_по_умолчанию).
# Примеры: "100 г" -> (mass, 0.1), "уп 10 шт" -> (pieces, 10), "2*500 г" -> (mass, 1.0), "тыс. шт" -> (pieces, 1000).
# custom - пользовательские правила {слово: (группа, количество)}: они главнее встроенных единиц.
# Если количество определить нельзя (голое "уп", незнакомая единица), группа и количество - None
def parse_unit(text, custom: dict = None) -> tuple:
    custom = custom or {}
    is_pack, found, rule = False, None, None

    for amount, word in UNIT_PART_RE.findall(expand_amounts(text).strip()):
        if word in custom:
            rule = rule or (amount, word)
            is_pack = is_pack or word.startswith(PACK_PREFIXES)
        elif word in UNITS:
            found = found or (amount, word)
        elif word.startswith(PACK_PREFIXES):
            is_pack = True

    # Явная единица в ячейке главнее правила. "уп шт" без числа - сколько штук в упаковке, из единицы не понять
    if found and not (is_pack and not found[0]):
        amount, word = found
        group, factor = UNITS[word]
        base = float(amount or 1) * factor

        return (group, base, is_pack, False) if base > 0 else (None, None, is_pack, False)

    # Пользовательское правило ("бухта = 100 м"): для упаковок это значение по умолчанию (название может его уточнить)
    if rule:
        amount, word = rule
        group, factor = custom[word]

        return group, float(amount or 1) * factor, is_pack, True

    return None, None, is_pack, False


# Похож ли текст на единицу измерения целиком: все слова - известные единицы, упаковки или правила пользователя.
# "100 г", "уп 10 шт", "тыс. шт" - да; "Мука 1 кг" - нет (есть постороннее слово). Нужен мапперу для определения колонки unit
def is_unit_text(text, custom: dict = None) -> bool:
    words = re.findall(r"[а-яa-z]+", expand_amounts(text))

    return bool(words) and all(w in UNITS or w in (custom or {}) or w == "по" or w.startswith(PACK_PREFIXES)
                               for w in words)


# Содержимое упаковки из названия товара: "Гвозди уп. по 50 шт" -> (pieces, 50). Не нашли - (None, None)
def parse_pack_from_name(name: str) -> tuple:
    text = expand_amounts(name)

    for regex in (NAME_PACK_RE, NAME_AMOUNT_RE):
        match = regex.search(text)

        if match and match.group(2) in UNITS:
            group, factor = UNITS[match.group(2)]
            base = float(match.group(1)) * factor

            if base > 0:
                return group, base

    return None, None


# Главная функция
# ----------------------------------------------------------------------------------------------------------------------

# Просит ИИ разобрать единицы, которые не поддались правилам. items: [(ключ, единица, название товара)].
# Контракт: в ИИ уходит {"items": [{"id", "unit", "name"}]}, в ответ ждем {"results": [{"id", "amount", "base_unit"}]}.
# Возвращает {ключ: (группа, количество в базовых единицах)} только для уверенных ответов; ИИ недоступен -> {}
def ai_resolve_units(items: list) -> dict:
    resolved, pending = {}, []

    for key, unit, name in items:
        if key in _ai_cache:
            if _ai_cache[key]:
                resolved[key] = _ai_cache[key]
        else:
            pending.append((key, unit, name))

    instruction = (
        "Ты помогаешь нормализовать единицы измерения прайс-листов. Для каждой позиции определи, сколько базовых "
        "единиц содержит ОДНА продаваемая единица unit (учитывай название товара name). Допустимые base_unit: "
        f"{', '.join(sorted(UNITS))}. Примеры: 'уп' + 'Гвозди уп. по 50 шт' -> 50 шт; 'бухта' + 'Провод 100 м' -> "
        "100 м; 'тыс. шт' -> 1000 шт. Если не уверен - amount: null, не угадывай. "
        "Формат ответа: {\"results\": [{\"id\": 0, \"amount\": 50, \"base_unit\": \"шт\"}]}."
    )

    pending = pending[:AI_MAX_ITEMS]  # ограничиваем стоимость и задержку одного прогона
    for start in range(0, len(pending), AI_BATCH_SIZE):
        batch = pending[start:start + AI_BATCH_SIZE]
        payload = {"items": [{"id": i, "unit": unit, "name": name} for i, (_, unit, name) in enumerate(batch)]}
        answer = ai_client.request_json("resolve_units", instruction, payload, expect=dict)
        results = answer.get("results") if answer else None
        if not isinstance(results, list):
            continue  # ИИ недоступен или ответ непригоден - строки останутся без пересчета

        parsed = {}
        for item in results:
            item = item if isinstance(item, dict) else {}
            amount, base_unit = item.get("amount"), item.get("base_unit")
            valid = isinstance(amount, (int, float)) and not isinstance(amount, bool) and \
                AI_AMOUNT_RANGE[0] <= amount <= AI_AMOUNT_RANGE[1] and base_unit in UNITS

            if valid and isinstance(item.get("id"), int):
                group, factor = UNITS[base_unit]
                parsed[item["id"]] = (group, amount * factor)

        # Запоминаем ответ (в том числе "не знаю"), чтобы не спрашивать ИИ про то же в следующих таблицах
        for i, (key, _, _) in enumerate(batch):
            if len(_ai_cache) < MAX_AI_CACHE:
                _ai_cache[key] = parsed.get(i)
            if i in parsed:
                resolved[key] = parsed[i]

    return resolved


# Коэффициент пересчета цены из валюты src в целевую dst
def rate_factor(src: str, dst: str, rates: dict) -> float:
    if src not in rates:
        log.warning("Нет курса для валюты %s, цены не пересчитываем", src)
        return 1.0

    return rates[src] / rates[dst]


# Приводит цены к целевой валюте, а количества и цены - к целевым единицам из конфигуратора.
# df: колонки price, qty, unit, name (все необязательные; необязательна и currency - валюта каждой строки).
# price_col_header: заголовок исходной колонки цены ("Цена, $") - по нему определяем валюту, если ее нет в данных.
# Возвращает копию df: price/qty/unit пересчитаны, добавлена колонка currency с целевой валютой
def normalize_units_and_currency(df: pd.DataFrame, config_df: pd.DataFrame = None, price_col_header: str = "") -> pd.DataFrame:
    config = parse_config(config_df)
    data = df.copy().reset_index(drop=True)  # позиционный индекс, исходный вернем в конце
    # --- Валюта: векторное умножение на курс; коэффициент считаем только по уникальным валютам
    price = pd.to_numeric(data["price"], errors="coerce") if "price" in data else None
    qty = pd.to_numeric(data["qty"], errors="coerce") if "qty" in data else None
    header_currency = detect_currency(price_col_header)
    target = config["currency"]

    if price is not None:
        if "currency" in data:
            raw = data["currency"].fillna("").astype(str)
            codes = raw.map({v: detect_currency(v) if v.strip() else header_currency for v in raw.unique()})
        else:
            codes = pd.Series(header_currency, index=data.index)

        # Курсы нужны, только если есть что пересчитывать: цены уже в целевой валюте не требуют обращения в сеть
        if (codes != target).any():
            rates = get_rates(config["rates_url"])

            if target not in rates:
                log.warning("Нет курса для целевой валюты %s, используем %s", target, DEFAULT_CURRENCY)
                target = DEFAULT_CURRENCY

            price = price * codes.map({c: 1.0 if c == target else rate_factor(c, target, rates)
                                       for c in codes.unique()})

    # --- Единицы: разбираем только уникальные значения unit, дальше .map() и векторные операции
    if "unit" in data:
        units = data["unit"].fillna("").astype(str)
        names = data["name"].fillna("").astype(str) if "name" in data else pd.Series("", index=data.index)
        specs = {u: parse_unit(u, config["custom_units"]) for u in units.unique()}
        group = units.map({u: s[0] for u, s in specs.items()})
        base = units.map({u: s[1] for u, s in specs.items()}).astype("float64")
        is_pack = units.map({u: s[2] for u, s in specs.items()}).astype(bool)
        is_default = units.map({u: s[3] for u, s in specs.items()}).astype(bool)

        # Упаковка без количества ("уп") или с количеством "по умолчанию" из правила пользователя:
        # уточняем по названию товара (тоже по уникальным названиям)
        need_name = is_pack & (base.isna() | is_default)
        if need_name.any():
            parsed = {n: parse_pack_from_name(n) for n in names[need_name].unique()}
            found_group = names[need_name].map({n: p[0] for n, p in parsed.items()})
            found_base = names[need_name].map({n: p[1] for n, p in parsed.items()}).astype("float64")
            hit = found_base.index[found_base.notna()]
            group.loc[hit], base.loc[hit] = found_group[hit], found_base[hit]

        # Остальное, что не поддалось правилам, отдаем ИИ. Ключ: для упаковки - единица+название, иначе - единица
        unresolved = base.isna() & (units != "")
        if unresolved.any() and ai_client.is_enabled():
            keys = units.where(~is_pack, units + "\x01" + names)
            sub = pd.DataFrame({"key": keys, "unit": units, "name": names})[unresolved].drop_duplicates("key")
            answers = ai_resolve_units(list(zip(sub["key"], sub["unit"], sub["name"])))

            if answers:
                ai_group = keys[unresolved].map({k: v[0] for k, v in answers.items()})
                ai_base = keys[unresolved].map({k: v[1] for k, v in answers.items()}).astype("float64")
                hit = ai_base.index[ai_base.notna()]
                group.loc[hit], base.loc[hit] = ai_group[hit], ai_base[hit]
                log.info("ИИ помог разобрать единицы в %d строках", len(hit))

        # Сколько целевых единиц в одной продаваемой: 100 г -> кг = 0.1; уп 10 шт -> шт = 10
        target_factor = group.map({g: UNITS[u][1] for g, u in config["targets"].items()}).astype("float64")
        ratio = base / target_factor
        valid = ratio.notna() & (ratio > 0)

        # Цена за целевую единицу обратно пропорциональна количеству, остаток - прямо пропорционален
        if price is not None:
            price = (price / ratio).where(valid, price)
        if qty is not None:
            qty = (qty * ratio).where(valid, qty)

        data["unit"] = group.map(config["targets"]).where(valid, data["unit"])

    # Округление убирает артефакты float (0.30000000000000004)
    if price is not None:
        data["price"] = price.round(6)
    if qty is not None:
        data["qty"] = qty.round(6)

    data["currency"] = target
    data.index = df.index

    return data
