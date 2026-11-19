import re
import difflib
import logging
import pandas as pd

import ai_client
import normalizer

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Константы алгоритма
# ----------------------------------------------------------------------------------------------------------------------

SAMPLE_SIZE = 30  # сколько непустых значений колонки смотрим при определении роли
MATCH_SHARE = 0.8  # какая доля значений выборки должна подойти под эвристику
LLM_EXAMPLES = 3  # сколько примеров значений колонки отправляем в LLM

ROLES = ("barcode", "sku", "price", "qty", "unit", "name", "other")
NAME_HEADER_WORDS = ("назван", "наимен", "товар", "номенкл", "product", "name")
BARCODE_HEADER_WORDS = ("штрихкод", "ean", "barcode", "gtin")
OTHER_PREFIX = "__o"  # служебный префикс колонок "other" внутри таблицы (сохраняем оригинальное имя)
IMPORTANT_ROLES = ("name", "price")  # если их не нашли эвристиками - зовем LLM-помощника

EAN_LENGTHS = (8, 12, 13)  # EAN-8, UPC-A, EAN-13
UNIT_WORDS = {"шт", "штук", "уп", "упак", "кг", "г", "гр", "м", "см", "мм", "л", "мл", "компл", "комплект", "рул",
              "пач"}
UNIT_HEADER_WORDS = ("ед.", "ед ", "единиц", "unit", "изм")  # маркеры колонки единиц в заголовке
UNIT_HEADER_SHARE = 0.5  # порог доли известных единиц, если заголовок это подтверждает
UNIT_MAX_UNIQUE = 12  # у колонки единиц измерения низкая уникальность
DEFAULT_UNIT = "шт"

# Маркеры в заголовках колонок (помогают отличить цену от остатка, когда в значениях одни целые числа)
PRICE_HEADER_WORDS = ("цена", "цены", "стоимость", "price", "руб", "прайс", "cost")
QTY_HEADER_WORDS = ("остаток", "остатки", "кол-во", "колич", "наличие", "qty", "stock", "склад")
CURRENCY_RE = re.compile(r"[₽$€]|руб|rub|usd|eur", re.IGNORECASE)

# Складские маркеры -> числовой остаток
QTY_MARKERS = {"много": 10, "в наличии": 10, "есть": 10, "+": 10, "мало": 1, "под заказ": 0, "нет": 0, "-": 0}

DECIMAL_RE = re.compile(r"^\d+[.,]\d{1,2}$")
INT_RE = re.compile(r"^\d+(?:\.0+)?$")

# Пороги сравнения названий
SIM_SAME = 0.90  # классическое сходство: выше - один товар (если числа в названиях не расходятся)
SIM_DIFFERENT = 0.65  # ниже - разные товары (если и вхождение слов мало)
SIM_FALLBACK = 0.80  # резерв без ИИ для серой зоны по сходству
CONTAINMENT_GRAY = 0.70  # доля слов короткого названия в длинном: выше - спорная пара
CONTAINMENT_FALLBACK = 0.85  # резерв без ИИ: короткое название почти целиком есть в длинном
MIN_CONTAINMENT_CHARS = 5  # по вхождению склеиваем, только если в коротком названии есть слово от 5 букв ("болт" - нет)
MAX_BUCKET = 150  # корзина больше этого - слово слишком частое, по нему не блокируем
LLM_BATCH_SIZE = 25  # сколько спорных пар отправляем в LLM за один запрос
# Разные числа в названиях ("Болт 10х50" и "Болт 10х60") - не склеиваем автоматически, решает только арбитр
STRICT_NUMBERS = True

# Служебные слова: не участвуют в блокировке (но участвуют в проверке вхождения)
STOP_WORDS = {"для", "без", "при", "над", "под", "шт", "уп", "кг", "мм", "см", "мл", "компл", "рул", "пач",
              "с", "и", "в", "пк", "компьютер", "портативный", "персональный"}

# Бренды и товарные слова: кириллица -> латиница (в дополнение к посимвольной транслитерации)
BRAND_MAP = {"макбук": "macbook", "айфон": "iphone", "самсунг": "samsung", "бош": "bosch", "эпл": "apple"}
TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya"}

# Итоговые колонки мастер-таблицы
MASTER_COLS = ["Наименование", "Штрихкод", "Цены", "Поставщики", "Количество", "Артикулы", "Ед. изм."]
OFFER_SEP = " | "  # разделитель списков предложений товара (позиции в списках совпадают между собой)
NO_VALUE = "—"  # прочерк на месте отсутствующего значения, чтобы не сбить позиционное соответствие


# ----------------------------------------------------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------------------------------------------------

# Пустое ли значение (None, NaN, пустая строка, "nan")
def is_empty(value) -> bool:
    if value is None:
        return True

    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0

    if pd.isna(value):
        return True

    # Прочерки и "н/д" - заглушки пустых ячеек (одиночный "-" не трогаем: это маркер нулевого остатка)
    return str(value).strip().lower() in ("", "nan", "none", "null", "—", "–", "н/д", "n/a")


# Значение ячейки -> чистая строка (списки склеиваем через запятую)
def to_text(value) -> str:
    if is_empty(value):
        return ""

    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value).strip()

    return str(value).replace("\xa0", " ").strip()


# Проверка контрольной суммы EAN-8 / UPC-A / EAN-13 (Модуль 10): веса 3 и 1 справа налево, не считая последнюю цифру
def is_valid_ean(code: str) -> bool:
    if not code.isdigit() or len(code) not in EAN_LENGTHS:
        return False

    body, check = code[:-1], int(code[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))

    return (10 - total % 10) % 10 == check


# Приводит значение к чистому штрихкоду: убирает пробелы и хвост ".0" (так числа приходят из pandas), иначе ""
def clean_ean(value) -> str:
    code = re.sub(r"[\s-]", "", to_text(value))
    code = re.sub(r"\.0+$", "", code)

    if not is_valid_ean(code):
        return ""

    # UPC-A (12 цифр) - это тот же EAN-13 с ведущим нулем: приводим к одному виду для поиска дублей
    return "0" + code if len(code) == 12 else code


# Цена -> float: убираем пробелы и валюту, запятую меняем на точку. Не получилось - None
def parse_price(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if pd.isna(value) else float(value)

    text = re.sub(r"[^\d.,-]", "", to_text(value))
    text = text.replace(",", ".").strip(".")  # крайние точки остаются от "руб." и подобного

    # Несколько точек ("1.250.50") - все, кроме последней, это разделители тысяч
    if text.count(".") > 1:
        head, _, tail = text.rpartition(".")
        text = head.replace(".", "") + "." + tail

    try:
        return float(text)
    except ValueError:
        return None


# Остаток -> int: складские маркеры переводим в числа, остальное - первое целое число из строки
def parse_qty(value) -> int:
    text = to_text(value).lower()

    if text in QTY_MARKERS:
        return QTY_MARKERS[text]

    match = re.search(r"\d+", text.replace(" ", ""))
    return int(match.group()) if match else 0


# Нормализованное название для поиска дублей (Token Sort): нижний регистр, только буквы и цифры, слова по алфавиту
def normalize_name(value) -> str:
    text = to_text(value).lower().replace("ё", "е")
    return " ".join(sorted(re.findall(r"[a-zа-я0-9]+", text)))



# ----------------------------------------------------------------------------------------------------------------------
# Основной класс
# ----------------------------------------------------------------------------------------------------------------------

# Посимвольная транслитерация кириллицы: "макбук" -> "makbuk"
def transliterate(word: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in word)


# Все написания слова, по которым его можно найти в другом названии: "макбук" -> {макбук, makbuk, macbook}
def word_forms(word: str) -> set:
    forms = {word}
    if re.search(r"[а-я]", word):
        forms.add(transliterate(word))
    if word in BRAND_MAP:
        forms.add(BRAND_MAP[word])
    return forms


# Профиль названия (из _name_norm): список (слово, его написания) и объединение всех написаний
def name_profile(name_norm: str) -> tuple:
    tokens = [(w, word_forms(w)) for w in name_norm.split()]
    forms = set().union(*(f for _, f in tokens)) if tokens else set()
    return tokens, forms


# Ключи блокировки: все написания значимых слов (от 4 букв, без служебных). Строка попадает в корзину по каждому
def blocking_keys(profile: tuple) -> set:
    keys = set()
    for word, forms in profile[0]:
        if len(word) >= 4 and word not in STOP_WORDS:
            keys |= forms
    return keys


# Доля слов короткого названия, которые встречаются в длинном ("Макбук" в "...apple macbook 16 гб" -> 1.0).
# Короткое - то, в котором меньше слов. Короткие слова ("болт") по вхождению не учитываем
def containment(pa: tuple, pb: tuple) -> float:
    short, long_ = (pa, pb) if len(pa[0]) <= len(pb[0]) else (pb, pa)
    tokens = short[0]
    if not tokens or max(len(w) for w, _ in tokens) < MIN_CONTAINMENT_CHARS:
        return 0.0

    hits = sum(1 for _, forms in tokens if forms & long_[1])
    return hits / len(tokens)


# Различаются ли числа в двух названиях, когда оба их содержат ("10х50" и "10х60"). Если в одном чисел нет - не спорим
def numbers_conflict(a: str, b: str) -> bool:
    na, nb = re.findall(r"\d+", a), re.findall(r"\d+", b)
    return STRICT_NUMBERS and bool(na and nb) and sorted(na) != sorted(nb)


class TableMapper:
    NUMERIC_RE = re.compile(r"^[₽$€]?\s*-?\d[\d\s.,]*\s*(?:₽|\$|€|руб\.?|р\.?|rub|usd|eur)?$", re.IGNORECASE)
    SKU_HEADER_WORDS = ("артикул", "арт.", "sku", "article")
    UNIT_ALIASES = {"штук": "шт", "упак": "уп", "гр": "г", "комплект": "компл"}
    HEADER_MARKERS = {"price": PRICE_HEADER_WORDS, "qty": QTY_HEADER_WORDS, "sku": SKU_HEADER_WORDS,
                      "barcode": BARCODE_HEADER_WORDS}

    def __init__(self, use_llm: bool = True, config_df: pd.DataFrame = None):
        # Без ключа LLM не вызываем вообще: все решения принимают эвристики и резервный порог
        self.__use_llm = use_llm and ai_client.is_enabled()

        # Конфигуратор из MWS Tables: цели перевода единиц/валюты и пользовательские правила для normalizer
        self.__config_df = config_df
        self.__custom_units = normalizer.parse_config(config_df)["custom_units"]

    # ------------------------------------------------------------------------------------------------------------------
    # Этап 1. Канонизация колонок: 7 эвристик (бесплатно) + LLM-помощник для оставшихся колонок
    # ------------------------------------------------------------------------------------------------------------------

    # Выборка до 30 непустых значений колонки (строками)
    def __sample(self, series: pd.Series) -> list:
        values = (to_text(v) for v in series)
        return [v for v in values if v][:SAMPLE_SIZE]

    # Доля значений выборки, для которых check() вернул True
    def __share(self, values: list, check) -> float:
        return sum(1 for v in values if check(v)) / len(values)

    # 1. Штрихкод: EAN-8 / UPC / EAN-13 с проверкой контрольной суммы. Проверяется первым и побеждает даже заголовок "шт"
    def __is_barcode(self, values: list, header: str) -> bool:
        return self.__share(values, lambda v: clean_ean(v) != "") >= MATCH_SHARE

    # 2. Единицы измерения: значения из словаря и мало уникальных
    def __is_unit(self, values: list, header: str) -> bool:
        in_dict = self.__share(values, lambda v: v.lower().strip(". ") in UNIT_WORDS
                               or normalizer.is_unit_text(v, self.__custom_units))  # "100 г", "уп 10 шт", "бухта"

        # Если заголовок говорит "Ед."/"Единица"/"Unit", в колонке могут быть и незнакомые единицы (их разберет normalizer)
        has_unit_header = header.startswith("ед") or any(w in header for w in UNIT_HEADER_WORDS)
        threshold = UNIT_HEADER_SHARE if has_unit_header else MATCH_SHARE
        return in_dict >= threshold and len(set(values)) <= UNIT_MAX_UNIQUE

    # Даты отдельной ролью не выделяем: такие колонки остаются "other" и сохраняются в сводке под своим именем

    # 4. Цена: дробные числа с 1-2 знаками, символы валют или маркер цены в заголовке
    def __is_price(self, values: list, header: str) -> bool:
        if self.__share(values, lambda v: bool(self.NUMERIC_RE.match(v))) < MATCH_SHARE:
            return False

        has_price_header = any(w in header for w in PRICE_HEADER_WORDS)
        if any(w in header for w in QTY_HEADER_WORDS) and not has_price_header:
            return False

        currency = self.__share(values, lambda v: bool(CURRENCY_RE.search(v))) >= 0.6
        decimals = self.__share(values, lambda v: bool(DECIMAL_RE.match(re.sub(r"\s", "", v)))) >= 0.6

        return currency or decimals or has_price_header

    # 5. Остатки: целые неотрицательные числа или складские маркеры ("в наличии", "нет", "+" ...)
    def __is_qty(self, values: list, header: str) -> bool:
        is_marker = lambda v: v.lower() in QTY_MARKERS
        is_int = lambda v: bool(INT_RE.match(v))

        # Заголовок про остатки ("Остаток", "Кол-во") допускает мусор в части ячеек ("гойда") - порог ниже
        has_qty_header = any(w in header for w in QTY_HEADER_WORDS)
        if self.__share(values, lambda v: is_marker(v) or is_int(v)) < (UNIT_HEADER_SHARE if has_qty_header else MATCH_SHARE):
            return False

        # Есть хотя бы один текстовый маркер или заголовок про остатки - этого достаточно
        if any(is_marker(v) for v in values) or any(w in header for w in QTY_HEADER_WORDS):
            return True

        # Просто целые числа: отсекаем длинные уникальные коды (это скорее артикулы) и колонки с заголовком про цену
        if any(w in header for w in PRICE_HEADER_WORDS):
            return False

        unique_ratio = len(set(values)) / len(values)
        avg_len = sum(len(v) for v in values) / len(values)
        return unique_ratio <= 0.5 or avg_len <= 3

    # 6. Артикул: 3-25 символов без пробелов, есть цифры и буквы/дефисы, уникальность > 50%
    def __is_sku(self, values: list, header: str) -> bool:
        def check(v: str) -> bool:
            if not 3 <= len(v) <= 25 or re.search(r"\s", v) or not re.search(r"\d", v):
                return False

            # Только цифры допускаем, когда заголовок прямо говорит, что это артикул
            return bool(re.search(r"[a-zа-я\-/._]", v, re.IGNORECASE)) or any(w in header for w in self.SKU_HEADER_WORDS)

        return self.__share(values, check) >= MATCH_SHARE and len(set(values)) / len(values) > 0.5

    # 7. Наименование: свободный текст (длинные значения с буквами, обычно с пробелами)
    def __is_name(self, values: list, header: str) -> bool:
        with_letters = self.__share(values, lambda v: bool(re.search(r"[a-zа-яё]", v, re.IGNORECASE)))
        with_spaces = self.__share(values, lambda v: " " in v)
        avg_len = sum(len(v) for v in values) / len(values)

        # Заголовок "Название"/"Товар" важнее длины: короткие наименования ("Макбук", "Мышь") тоже названия
        if any(w in header for w in NAME_HEADER_WORDS):
            return with_letters >= MATCH_SHARE

        return with_letters >= MATCH_SHARE and (with_spaces >= 0.5 or avg_len >= 8)

    # Определяет роль каждой колонки: список ролей в порядке колонок df. Колонки без роли - "other"
    def detect_roles(self, df: pd.DataFrame) -> list:
        # Порядок важен: штрихкод проверяется первым, наименование (самая "жадная" эвристика) - последним
        checks = [("barcode", self.__is_barcode), ("unit", self.__is_unit), ("price", self.__is_price),
                  ("qty", self.__is_qty), ("sku", self.__is_sku), ("name", self.__is_name)]
        headers = [str(c).strip().lower() for c in df.columns]
        samples = [self.__sample(df.iloc[:, i]) for i in range(df.shape[1])]

        # Шаг 1: первая подходящая эвристика. На одну роль может претендовать несколько колонок
        roles = [next((r for r, check in checks if values and check(values, headers[i])), "other")
                 for i, values in enumerate(samples)]

        # Шаг 2: конфликты (2 и более колонки на одну роль) разрешаем: ИИ или эвристика
        roles = self.__resolve_conflicts(df, roles, samples, headers)

        # Шаг 3: LLM-помощник для неразмеченных колонок, если не нашли name или price
        if self.__use_llm:
            roles = self.__llm_assign_roles(df, roles, samples)

        return roles

    # Конфликт ролей: на одну роль претендуют несколько колонок. Оставляем одну, остальные уходят в "other"
    # (их значения все равно сохранятся в сводке под своими именами)
    def __resolve_conflicts(self, df: pd.DataFrame, roles: list, samples: list, headers: list) -> list:
        conflicts = {}
        for role in set(roles) - {"other"}:
            indexes = [i for i, r in enumerate(roles) if r == role]
            if len(indexes) > 1:
                conflicts[role] = indexes

        if not conflicts:
            return list(roles)

        # ИИ выбирает основную колонку по заголовкам и примерам значений; без ИИ - эвристика
        chosen = self.__llm_pick_conflicts(df, conflicts, samples) if self.__use_llm else {}

        result = list(roles)
        for role, indexes in conflicts.items():
            keep = chosen.get(role, self.__fallback_pick(role, indexes, samples, headers))
            for i in indexes:
                if i != keep:
                    result[i] = "other"

        return result

    # Без ИИ: для name - самая длинная текстовая колонка; для остальных - колонка с подходящим заголовком, иначе первая
    def __fallback_pick(self, role: str, indexes: list, samples: list, headers: list) -> int:
        if role == "name":
            return max(indexes, key=lambda i: sum(len(v) for v in samples[i]) / len(samples[i]))

        markers = self.HEADER_MARKERS.get(role, ())
        return next((i for i in indexes if any(w in headers[i] for w in markers)), indexes[0])

    # ИИ выбирает одну колонку на каждую конфликтующую роль. Ответ не по контракту - пусто, сработает эвристика
    def __llm_pick_conflicts(self, df: pd.DataFrame, conflicts: dict, samples: list) -> dict:
        payload = {"conflicts": [
            {"role": role, "columns": [{"id": i, "header": str(df.columns[i]), "examples": samples[i][:LLM_EXAMPLES]}
                                       for i in indexes]}
            for role, indexes in conflicts.items()]}
        instruction = (
            "В прайс-листе несколько колонок претендуют на одну роль. Для каждой роли выбери ОДНУ колонку (id), "
            "которая действительно ее содержит. Формат ответа: {\"choice\": {\"<роль>\": <id колонки>}}."
        )
        answer = ai_client.request_json("resolve_column_conflicts", instruction, payload, expect=dict)
        choice = answer.get("choice") if answer else None
        if not isinstance(choice, dict):
            return {}

        result = {}
        for role, indexes in conflicts.items():
            value = choice.get(role)
            if str(value).isdigit() and int(value) in indexes:
                result[role] = int(value)

        return result

    # LLM-помощник: одним запросом отправляем названия и 3 примера значений неразмеченных колонок
    def __llm_assign_roles(self, df: pd.DataFrame, roles: list, samples: list) -> list:
        missing = [r for r in IMPORTANT_ROLES if r not in roles]
        candidates = [i for i, r in enumerate(roles) if r == "other" and samples[i]]

        if not missing or not candidates:
            return roles

        columns = [{"id": i, "header": str(df.columns[i]), "examples": samples[i][:LLM_EXAMPLES]} for i in candidates]
        instruction = (
            "Ты помогаешь размечать колонки прайс-листов. Возможные роли: "
            "name (наименование товара), price (цена), sku (артикул), qty (остаток), unit (единица измерения), "
            "barcode (штрихкод), other (не подходит). "
            "Формат ответа: {\"roles\": {\"<id колонки>\": \"<роль>\"}}."
        )
        answer = ai_client.request_json("detect_column_roles", instruction, {"columns": columns}, expect=dict)
        roles_answer = answer.get("roles") if answer else None
        if not isinstance(roles_answer, dict):
            return roles

        result = list(roles)
        for key, role in roles_answer.items():
            # Принимаем только известные роли для запрошенных колонок и не создаем дубли уже найденных ролей
            if str(key).isdigit() and int(key) in candidates and role in ROLES and role != "other" \
                    and role not in result:
                result[int(key)] = role
                log.info("LLM определила колонку '%s' как '%s'", df.columns[int(key)], role)

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Этап 2. Очистка и нормализация значений
    # ------------------------------------------------------------------------------------------------------------------

    # Приводит сырую таблицу к стандартному виду: _name, _sku, _price, _qty, _unit + ключи _ean и _name_norm
    def standardize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        roles = self.detect_roles(df)
        cols = {role: df.iloc[:, roles.index(role)] for role in set(roles) - {"other"}}
        empty = pd.Series([None] * len(df), index=df.index)
        get = lambda role: cols.get(role, empty)

        std = pd.DataFrame({
            "_source": source,
            "_name": [to_text(v) for v in get("name")],
            "_sku": [to_text(v) for v in get("sku")],
            "_price": [parse_price(v) if not is_empty(v) else None for v in get("price")],
            # Валюта из самой ячейки ("15.50$"): parse_price символ отбрасывает, поэтому запоминаем его отдельно
            "_currency": [normalizer.find_currency(to_text(v)) for v in get("price")],
            "_qty": [parse_qty(v) for v in get("qty")],
            "_unit": [self.__normalize_unit(v) for v in get("unit")],
            "_ean": [clean_ean(v) for v in get("barcode")]
        }, index=df.index)
        std["_name_norm"] = [normalize_name(v) for v in std["_name"]]

        # Колонки "other" не выбрасываем: значения сохраняются под оригинальным именем (ключ - номер колонки + имя)
        for i, role in enumerate(roles):
            if role == "other":
                std[f"{OTHER_PREFIX}{i}|{df.columns[i]}"] = [to_text(v) for v in df.iloc[:, i]]

        # Валюты и единицы приводим к целевым ДО поиска дублей и минимальной цены. Валюту определяет заголовок
        # колонки цены именно этой таблицы
        price_header = str(df.columns[roles.index("price")]) if "price" in roles else ""
        std = self.__normalize_values(std, price_header)

        # Строки без названия, штрихкода и артикула идентифицировать нечем - отбрасываем
        keep = (std["_name"] != "") | (std["_ean"] != "") | (std["_sku"] != "")
        log.info("'%s': роли колонок %s, строк: %d", source, dict(zip(map(str, df.columns), roles)), int(keep.sum()))

        return std[keep].reset_index(drop=True)

    # Пересчет цен, остатков и единиц через normalizer: _price/_qty/_unit заменяются значениями в целевых валюте и единицах
    def __normalize_values(self, std: pd.DataFrame, price_header: str) -> pd.DataFrame:
        if std.empty:
            return std

        view = std.rename(columns={"_price": "price", "_qty": "qty", "_unit": "unit", "_name": "name",
                                   "_currency": "currency"})
        result = normalizer.normalize_units_and_currency(view[["price", "qty", "unit", "name", "currency"]],
                                                         self.__config_df, price_header)
        std = std.copy()
        std["_price"], std["_qty"], std["_unit"] = result["price"], result["qty"], result["unit"]

        return std

    # Единица измерения: нижний регистр, без точки, синонимы -> короткая форма
    def __normalize_unit(self, value) -> str:
        unit = to_text(value).lower().strip(". ")
        return self.UNIT_ALIASES.get(unit, unit)

    # ------------------------------------------------------------------------------------------------------------------
    # Этап 3. Каскадный мерджинг строк. ПО АРТИКУЛУ (sku) НЕ СКЛЕИВАЕМ: у разных поставщиков артикулы пересекаются
    # ------------------------------------------------------------------------------------------------------------------

    # Возвращает номер группы для каждой строки all_rows (строки одного товара получают одинаковый номер)
    def merge_groups(self, all_rows: pd.DataFrame) -> list:
        parent = list(range(len(all_rows)))

        # Union-Find: find со сжатием путей и union двух строк
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            parent[find(a)] = find(b)

        eans = all_rows["_ean"].tolist()
        names = all_rows["_name_norm"].tolist()

        # Ступень 1. Жесткий ключ O(N): склеиваем только по валидному штрихкоду
        first_by_ean = {}
        for i, ean in enumerate(eans):
            if ean:
                union(i, first_by_ean.setdefault(ean, i))

        # Ступень 2. Мульти-блокировка: строка без штрихкода попадает в корзину по каждому написанию каждого
        # значимого слова. Так "Макбук" и "...apple macbook..." встречаются в корзине "macbook"
        profiles = [name_profile(n) for n in names]
        buckets = {}
        for i in range(len(all_rows)):
            if not eans[i] and names[i]:
                for key in blocking_keys(profiles[i]):
                    buckets.setdefault(key, []).append(i)

        # Ступень 3. Сравнение внутри корзин. Пару сравниваем один раз, даже если она попала в несколько корзин
        compared = set()
        gray_pairs = []
        for rows in buckets.values():
            if len(rows) < 2 or len(rows) > MAX_BUCKET:
                continue  # одиночка не с кем сравнивается, а слишком частое слово не дает блокировки

            for pos, i in enumerate(rows):
                for j in rows[pos + 1:]:
                    if (i, j) in compared:
                        continue
                    compared.add((i, j))

                    cont = containment(profiles[i], profiles[j])
                    matcher = difflib.SequenceMatcher(None, names[i], names[j])

                    # Быстрая отсечка: quick_ratio - верхняя граница сходства. Если она мала и слова не вложены - пара пустая
                    if cont < CONTAINMENT_GRAY and matcher.quick_ratio() < SIM_DIFFERENT:
                        continue

                    sim = matcher.ratio()
                    if sim >= SIM_SAME and not numbers_conflict(names[i], names[j]):
                        union(i, j)
                    elif sim >= SIM_DIFFERENT or cont >= CONTAINMENT_GRAY:
                        gray_pairs.append((i, j, sim, cont))  # серая зона - отдадим арбитру

        # Ступень 4. Пакетный LLM-арбитр (с резервным порогом) для спорных пар
        gray_pairs = [p for p in gray_pairs if find(p[0]) != find(p[1])]
        for (i, j, _, _), same in zip(gray_pairs, self.__arbitrate(all_rows, gray_pairs)):
            if same:
                union(i, j)

        # Нумеруем группы в порядке первого появления
        group_ids = {}
        return [group_ids.setdefault(find(i), len(group_ids)) for i in range(len(all_rows))]

    # Различаются ли наборы чисел в двух нормализованных названиях (10 и 50 против 10 и 60)
    def __numbers_differ(self, a: str, b: str) -> bool:
        return numbers_conflict(a, b)

    # Решает спорные пары: список True/False. Пачками по LLM_BATCH_SIZE; при любой проблеме - порог SIM_FALLBACK
    def __arbitrate(self, all_rows: pd.DataFrame, pairs: list) -> list:
        # Резерв без ИИ: похожие названия или короткое почти полностью вложено в длинное, но без расходящихся чисел
        decisions = [(sim >= SIM_FALLBACK or cont >= CONTAINMENT_FALLBACK)
                     and not self.__numbers_differ(all_rows.at[i, "_name_norm"], all_rows.at[j, "_name_norm"])
                     for i, j, sim, cont in pairs]
        if not pairs or not self.__use_llm:
            return decisions

        instruction = (
            "Ты эксперт по товарам. Для каждой пары a/b реши, один ли это товар. Учитывай модель, размеры, "
            "характеристики. Формат ответа: {\"same\": [1 или 0, ...]} - по одному значению на пару, в том же "
            "порядке (1 - один товар, 0 - разные)."
        )
        describe = lambda i: {"name": all_rows.at[i, "_name"], "sku": all_rows.at[i, "_sku"],
                              "price": all_rows.at[i, "_price"] if pd.notna(all_rows.at[i, "_price"]) else None}

        for start in range(0, len(pairs), LLM_BATCH_SIZE):
            batch = pairs[start:start + LLM_BATCH_SIZE]
            payload = {"pairs": [{"a": describe(i), "b": describe(j)} for i, j, _, _ in batch]}
            answer = ai_client.request_json("match_products", instruction, payload, expect=dict)
            same = answer.get("same") if answer else None

            # Ответ принимаем только если это список нужной длины из 0/1, иначе остается резервный порог
            if isinstance(same, list) and len(same) == len(batch) and all(a in (0, 1) for a in same):
                decisions[start:start + len(batch)] = [a == 1 for a in same]

        return decisions

    # ------------------------------------------------------------------------------------------------------------------
    # Этап 4. Агрегация: каждая группа -> одна мастер-строка
    # ------------------------------------------------------------------------------------------------------------------

    # Сборка мастер-строки. Предложения одного товара сортируем по возрастанию цены (без цены - в конец).
    # Сортировка стабильная, поэтому все списки сводки совпадают по позициям: i-я цена - это i-й поставщик,
    # i-е количество, артикул, единица и колонка other
    def aggregate(self, all_rows: pd.DataFrame, groups: list) -> pd.DataFrame:
        all_rows = all_rows.assign(_group=groups)

        # Колонки "other" тоже идут по предложениям. Колонки, которых не было в таблице поставщика, - пустые
        other_keys = [c for c in all_rows.columns if c.startswith(OTHER_PREFIX)]
        labels = list(dict.fromkeys(output_label(k) for k in other_keys))
        all_rows = all_rows.fillna({k: "" for k in other_keys})
        result = []

        for _, g in all_rows.groupby("_group", sort=True):
            g = g.sort_values("_price", na_position="last", kind="mergesort")
            names = [n for n in g["_name"] if n]
            eans = [e for e in g["_ean"] if e]

            row = {
                "Наименование": max(names, key=len) if names else "",  # самое подробное название
                "Штрихкод": eans[0] if eans else "",
                "Цены": join_offers(fmt_price(p) for p in g["_price"]),
                "Поставщики": join_offers(g["_source"]),
                "Количество": join_offers(fmt_num(q) for q in g["_qty"]),
                "Артикулы": join_offers(g["_sku"]),
                "Ед. изм.": join_offers(g["_unit"]),
            }

            for label in labels:
                keys = [k for k in other_keys if output_label(k) == label]
                cells = [next((g[k].iloc[pos] for k in keys if g[k].iloc[pos]), "") for pos in range(len(g))]
                row[label] = join_offers(cells)

            result.append(row)

        return pd.DataFrame(result, columns=MASTER_COLS + labels)

    # ------------------------------------------------------------------------------------------------------------------
    # Полный конвейер
    # ------------------------------------------------------------------------------------------------------------------

    def build(self, dfs_by_name: dict) -> pd.DataFrame:
        standardized = []

        for name, df in dfs_by_name.items():
            if df is None or df.empty:
                log.info("Таблица '%s' пустая, пропускаем", name)
                continue

            standardized.append(self.standardize(df, name))

        standardized = [s for s in standardized if not s.empty]
        if not standardized:
            return pd.DataFrame(columns=MASTER_COLS)

        all_rows = pd.concat(standardized, ignore_index=True)
        groups = self.merge_groups(all_rows)
        master_df = self.aggregate(all_rows, groups)

        log.info("Строк до склейки: %d, товаров в сводке: %d", len(all_rows), len(master_df))
        return master_df


# Цена предложения для сводки: "1200.0", либо прочерк, если цены нет
def fmt_price(value) -> str:
    return NO_VALUE if value is None or pd.isna(value) else str(round(float(value), 4))  # мелкие цены за штуку не теряем


# Количество предложения для сводки: "250" (целое) или дробное, либо прочерк
def fmt_num(value) -> str:
    if value is None or pd.isna(value):
        return NO_VALUE
    value = float(value)
    return str(int(value)) if value.is_integer() else str(round(value, 3))


# Список предложений через OFFER_SEP; пустое значение - прочерк (позиция сохраняется)
def join_offers(items) -> str:
    return OFFER_SEP.join(str(x) if x not in (None, "") else NO_VALUE for x in items)


# Имя колонки "other" в сводке: оригинальное имя. Если оно совпало с системной колонкой - с пометкой
def output_label(key: str) -> str:
    label = key.partition("|")[2]
    return f"{label} (доп.)" if label in MASTER_COLS else label


# Функция-обертка для остального кода проекта
def build_master_table(dfs_by_name: dict, use_llm: bool = True, config_df: pd.DataFrame = None) -> pd.DataFrame:
    return TableMapper(use_llm=use_llm, config_df=config_df).build(dfs_by_name)

