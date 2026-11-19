import os
import re
import json
import time
import uuid
import logging
import threading
import requests

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Единственная точка работы с ИИ.
#
# Контракт (JSON in -> JSON out), одинаковый для всех провайдеров:
#   запрос  : request_json(task, instruction, payload)
#             в модель уходит сообщение {"task": "<имя задачи>", "input": <payload>}, а в system - instruction
#   ответ   : распарсенный JSON (dict или list), который модель вернула, либо None
#
# None означает "ИИ недоступен или ответ непригоден" (нет ключа, сеть, таймаут, не JSON, не тот тип).
# Вызывающий код ВСЕГДА обязан иметь резервный путь на случай None и сам проверять содержимое ответа.
#
# Провайдеры: "gigachat" (Сбер, бесплатный Freemium) и "openai" (любой OpenAI-совместимый API, например OpenRouter).
# ----------------------------------------------------------------------------------------------------------------------

# Секреты и настройки - только из переменных окружения системы
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "")  # "ключ авторизации" из личного кабинета (base64 client_id:secret)
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct:free")
# Если провайдер не указан явно: задан ключ GigaChat - используем его, иначе OpenAI-совместимый API
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("gigachat" if GIGACHAT_AUTH_KEY else "openai")
LLM_TIMEOUT = 60
LLM_MAX_TOKENS = 2048

# GigaChat. Freemium: scope GIGACHAT_API_PERS, модель GigaChat (Lite), генерация текста строго в ОДНОМ потоке
GIGACHAT_OAUTH_URL = os.getenv("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL", "https://api.giga.chat")  # с 17.07.2026; прежний gigachat.devices.sberbank.ru тоже работает
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2")  # Lite из Freemium; список доступных: GET /v1/models
GIGACHAT_CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE", "russian_trusted_root_ca.cer")  # путь к сертификатам Минцифры, если их нет в системе

# Токен доступа живет 30 минут. Обновляем заранее: за TOKEN_REFRESH_MARGIN до конца и в любом случае не позже
# TOKEN_MAX_AGE после получения (защита от неверного expires_at). Итого - чаще, чем раз в 30 минут
TOKEN_REFRESH_MARGIN = 300
TOKEN_MAX_AGE = 25 * 60
OAUTH_TIMEOUT = 10
GIGACHAT_RETRIES = 3  # попыток на один запрос: повтор после 401 (новый токен) и 429 (слишком часто)
GIGACHAT_QUEUE_TIMEOUT = 120  # сколько ждем своей очереди к модели (поток один)

JSON_ONLY_RULE = "Отвечай ТОЛЬКО валидным JSON без пояснений и markdown."

# Состояние только в памяти процесса (Zero storage): токен и его срок
_token = {"value": "", "refresh_at": 0.0}
_token_lock = threading.Lock()
_gigachat_slot = threading.Semaphore(1)  # Freemium: один запрос к модели одновременно


# Требования к TLS: сертификаты Минцифры можно указать файлом, иначе используется системное хранилище
def _verify():
    return GIGACHAT_CA_BUNDLE or True


# Время, когда токен пора менять: время в ответе OAuth бывает в секундах или в миллисекундах
def _refresh_at(expires_at, now: float) -> float:
    latest = now + TOKEN_MAX_AGE

    if isinstance(expires_at, (int, float)) and expires_at > 0:
        seconds = expires_at / 1000 if expires_at > 1e11 else expires_at
        return min(seconds - TOKEN_REFRESH_MARGIN, latest)

    return latest


# Запрашивает новый токен доступа (POST /api/v2/oauth). Ключ и токен в логи не пишем
def _fetch_token() -> bool:
    key = GIGACHAT_AUTH_KEY.strip()
    key = key[6:].strip() if key.lower().startswith("basic ") else key
    headers = {"Authorization": f"Basic {key}", "RqUID": str(uuid.uuid4()), "Accept": "application/json",
               "Content-Type": "application/x-www-form-urlencoded"}

    try:
        resp = requests.post(GIGACHAT_OAUTH_URL, headers=headers, data={"scope": GIGACHAT_SCOPE},
                             timeout=OAUTH_TIMEOUT, verify=_verify())
        resp.raise_for_status()
        data = resp.json()
        token = data["access_token"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        log.warning("GigaChat: не удалось получить токен доступа (%s)", type(e).__name__)
        return False

    _token["value"] = token
    _token["refresh_at"] = _refresh_at(data.get("expires_at"), time.time())
    return True


# Действующий токен доступа. Обновляется, когда подошел срок (или force=True после ответа 401).
# Замок гарантирует, что при параллельных запросах токен получит только один поток, остальные подождут его
def gigachat_token(force: bool = False) -> str:
    with _token_lock:
        if force or not _token["value"] or time.time() >= _token["refresh_at"]:
            if not _fetch_token():
                _token["value"] = ""

        return _token["value"]



# ИИ включен, только если задан ключ выбранного провайдера
def is_enabled() -> bool:
    return bool(GIGACHAT_AUTH_KEY) if LLM_PROVIDER == "gigachat" else bool(LLM_API_KEY)


# Адрес chat completions из LLM_API_URL (допускаем как корень сервиса, так и полный путь)
def build_endpoint(url: str) -> str:
    url = url.rstrip("/")

    if url.endswith("/chat/completions"):
        return url

    if re.search(r"/v\d+$", url):
        return f"{url}/chat/completions"

    if "openrouter.ai" in url and "/api" not in url:
        return f"{url}/api/v1/chat/completions"

    return f"{url}/v1/chat/completions"


# Достает JSON из текста ответа (модели любят оборачивать его в ```json ... ```). Не получилось - None
def extract_json(text, expect: type = None):
    if not text:
        return None

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start == -1 or end <= start:
            continue

        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            continue

        if expect is None or isinstance(data, expect):
            return data

    return None


# Ответ модели (текст) из тела Chat Completions
def _content(resp: requests.Response) -> str:
    return resp.json()["choices"][0]["message"]["content"]


# OpenAI-совместимый провайдер (OpenRouter и т.п.): статичный ключ в заголовке
def _chat_openai(messages: list) -> str:
    body = {"model": LLM_MODEL, "temperature": 0.0, "max_tokens": LLM_MAX_TOKENS, "messages": messages}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(build_endpoint(LLM_API_URL), json=body, headers=headers, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    return _content(resp)


# GigaChat: токен с автообновлением, запросы строго по одному (Freemium = один поток).
# 401 -> берем новый токен и повторяем; 429 -> короткая пауза и повтор; остальные ошибки - исключение
def _chat_gigachat(messages: list) -> str:
    body = {"model": GIGACHAT_MODEL, "temperature": 0.0, "max_tokens": LLM_MAX_TOKENS, "messages": messages}
    url = f"{GIGACHAT_API_URL.rstrip('/')}/v1/chat/completions"

    if not _gigachat_slot.acquire(timeout=GIGACHAT_QUEUE_TIMEOUT):
        raise requests.Timeout("очередь к GigaChat занята")

    try:
        force_token = False

        for attempt in range(GIGACHAT_RETRIES):
            token = gigachat_token(force=force_token)
            if not token:
                raise requests.ConnectionError("нет токена доступа")

            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                       "Accept": "application/json"}
            resp = requests.post(url, json=body, headers=headers, timeout=LLM_TIMEOUT, verify=_verify())
            force_token = resp.status_code == 401

            if force_token:
                continue

            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

            resp.raise_for_status()
            return _content(resp)

        raise requests.HTTPError("GigaChat: исчерпаны попытки (401/429)")
    finally:
        _gigachat_slot.release()


# Отправляет задачу в ИИ и возвращает JSON-ответ (или None). expect - ожидаемый тип верхнего уровня (dict / list)
def request_json(task: str, instruction: str, payload, expect: type = None):
    if not is_enabled():
        return None

    messages = [
        {"role": "system", "content": f"{instruction}\n{JSON_ONLY_RULE}"},
        {"role": "user", "content": json.dumps({"task": task, "input": payload}, ensure_ascii=False)}
    ]

    try:
        content = (_chat_gigachat if LLM_PROVIDER == "gigachat" else _chat_openai)(messages)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        log.warning("ИИ недоступен для задачи '%s' (%s%s)", task, type(e).__name__, f", HTTP {status}" if status else "")
        return None

    result = extract_json(content, expect)
    if result is None:
        # Показываем начало ответа: по нему видно, обрезан ли он лимитом токенов или это не JSON вовсе
        log.warning("ИИ вернул непригодный ответ для задачи '%s': %r", task, str(content)[:200])

    return result
