# Конфигурация gunicorn (подхватывается автоматически из рабочей директории)

bind = "127.0.0.1:8000"  # наружу не торчим: порт 80/443 держит nginx
worker_class = "gthread"
workers = 1  # счетчики rate limit хранятся в памяти процесса - несколько воркеров дали бы лимит x N
threads = 4  # параллельные запросы (в основном ожидание ответов MWS Tables / LLM)
timeout = 300  # конвейер с LLM и большими таблицами может идти долго
graceful_timeout = 30
keepalive = 5

# Лишний мусор от ботов обрываем на уровне сервера
limit_request_line = 2048
limit_request_fields = 50

# Логи в stdout/stderr -> journald (на диск ничего не пишем)
accesslog = "-"
errorlog = "-"
loglevel = "info"
