import os
import re
import hmac
import logging

import requests
from flask import Flask, request, jsonify, abort, current_app
from flask_limiter import Limiter
from flask_limiter.errors import RateLimitExceeded
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from mws_tables_client import MWSTablesClient
from pipeline import run_pipeline, PipelineError

TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]{10,128}")  # грубая проверка формата токена: бесплатно, без сети
MAX_BODY_SIZE = 4096  # тело запроса - несколько коротких строк, больше не принимаем

log = logging.getLogger(__name__)


# Текущий лимит из конфигурации приложения (например, "3 per minute")
def rate_limit() -> str:
    return current_app.config["RATE_LIMIT"]


# Пароль к /health в заголовке X-Health-Key. Сравнение за постоянное время; если пароль не задан - ручка закрыта для всех
def health_key_ok() -> bool:
    password = current_app.config["HEALTH_PASSWORD"]
    return bool(password) and hmac.compare_digest(request.headers.get("X-Health-Key", "").encode(), password.encode())


# Легкая проверка подлинности токена: проверка формата и один запрос GET /spaces.
# Выполняется ДО тяжелого конвейера (pandas), поэтому боты со случайными токенами до него не доходят.
# Возвращает список пространств токена - он же нужен для автопоиска пространства
def check_token(token: str) -> list:
    if not TOKEN_RE.fullmatch(token):
        abort(401, "Неверный токен")

    try:
        return MWSTablesClient(token=token).get_spaces()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            abort(401, "Неверный токен")

        abort(502, "MWS Tables вернул ошибку")
    except (requests.RequestException, RuntimeError):
        abort(502, "MWS Tables недоступен")


# Достает токен и необязательный spaceId из запроса (токен: тело JSON или заголовок Authorization: Bearer ...)
def parse_request() -> tuple:
    payload = request.get_json(force=True, silent=True)
    payload = payload if isinstance(payload, dict) else {}

    token = payload.get("token")
    if not token:
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else None

    space_id = payload.get("spaceId") or payload.get("space_id")

    if not isinstance(token, str) or not token:
        abort(401, "Токен не передан")

    if space_id is not None and (not isinstance(space_id, str) or len(space_id) > 64):
        abort(400, "Некорректный spaceId")

    return token, space_id


def create_app(overrides: dict = None) -> Flask:
    app = Flask(__name__)

    # Конфигурация и секреты - только из переменных окружения системы
    app.config.update(
        HEALTH_PASSWORD=os.getenv("HEALTH_PASSWORD", ""),
        RATE_LIMIT=f"{int(os.getenv('RATE_LIMIT_PER_MIN', '3'))} per minute",
        MAX_CONTENT_LENGTH=MAX_BODY_SIZE  # больше - Flask сам вернет 413, не читая тело
    )
    app.config.update(overrides or {})
    app.json.ensure_ascii = False  # русские сообщения без \u-экранирования

    # Логи в stdout (их собирает systemd/journald). Токены в логи не пишем
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # За nginx реальный IP клиента лежит в X-Forwarded-For. Доверяем ровно PROXY_HOPS прокси (по умолчанию 1 - nginx),
    # иначе заголовок можно подделать и обойти лимит. PROXY_HOPS=0 - без прокси
    hops = int(os.getenv("PROXY_HOPS", "1"))
    if hops:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=hops, x_proto=hops)

    # Лимит запросов: счетчики в памяти процесса (Zero storage), окно скользящее.
    # Поэтому gunicorn запускаем с ОДНИМ воркером (иначе лимит умножится на число воркеров)
    limiter = Limiter(get_remote_address, app=app, storage_uri="memory://", strategy="moving-window",
                      headers_enabled=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Ручки
    # ------------------------------------------------------------------------------------------------------------------

    # Основная ручка: ее вызывает Workflow из MWS Tables. Лимит проверяется Flask-Limiter ДО выполнения функции
    @app.post("/run")
    @limiter.shared_limit(rate_limit, scope="api")
    def run():
        token, space_id = parse_request()
        spaces = check_token(token)

        try:
            result = run_pipeline(token, space_id, spaces)
        except PipelineError as e:
            abort(e.status_code, e.message)
        except (requests.RequestException, RuntimeError) as e:
            log.error("Ошибка MWS Tables: %s", type(e).__name__)
            abort(502, "Ошибка при обращении к MWS Tables")

        return jsonify(status="ok", **result)

    # Закрытая ручка живости: без верного X-Health-Key притворяется несуществующей (404).
    # Лимит общий с /run (scope="api"): с одного IP не больше N запросов в минуту на все ручки.
    # Запросы с верным паролем (мониторинг) в лимит не входят, а подбор пароля упирается в него
    @app.get("/health")
    @limiter.shared_limit(rate_limit, scope="api", exempt_when=health_key_ok)
    def health():
        if not health_key_ok():
            abort(404)

        return jsonify(status="ok")

    # ------------------------------------------------------------------------------------------------------------------
    # Единый JSON-формат ошибок и заголовки
    # ------------------------------------------------------------------------------------------------------------------

    @app.errorhandler(HTTPException)
    def handle_http_error(e: HTTPException):
        message = "Слишком много запросов" if isinstance(e, RateLimitExceeded) else e.description
        if e.code in (404, 405):
            message = e.name  # короткое "Not Found" вместо длинного стандартного текста Werkzeug

        return jsonify(status="error", message=message), e.code

    # Любая непредвиденная ошибка: детали только в лог, клиенту - общий ответ
    @app.errorhandler(Exception)
    def handle_unexpected_error(e: Exception):
        log.exception("Необработанная ошибка")
        return jsonify(status="error", message="Внутренняя ошибка"), 500

    @app.after_request
    def set_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app
