import logging
import requests
from mws_tables_client import MWSTablesClient
from mapper import build_master_table

INPUT_FOLDER = "Input"  # папка с исходными таблицами (API не умеет создавать папки - она должна существовать)
OUTPUT_TABLE = "Output"  # итоговая таблица в корне пространства (если ее нет - создается)
# Таблица-конфигуратор в корне пространства: целевая валюта и единицы, свои правила перевода, ссылка на курсы
CONFIG_TABLES = ("Config", "Конфигуратор", "Настройки")

log = logging.getLogger(__name__)


# Читает конфигуратор из MWS Tables. Его отсутствие или сбой чтения - не ошибка: работаем с настройками по умолчанию
def load_config(client: MWSTablesClient, space_id: str):
    try:
        config_id = client.find_table(space_id, CONFIG_TABLES)
        if not config_id:
            log.info("Таблица-конфигуратор не найдена, используем настройки по умолчанию")
            return None

        config_df = client.read_df(config_id)
        log.info("Прочитан конфигуратор: %d строк", len(config_df))
        return config_df
    except (requests.RequestException, RuntimeError) as e:
        log.warning("Не удалось прочитать конфигуратор (%s), используем настройки по умолчанию", type(e).__name__)
        return None


# Ожидаемая ошибка пайплайна: несет HTTP-статус, который вернет сервер
class PipelineError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# Полный цикл Extract -> Transform -> Load. Ничего не хранит между вызовами (Zero storage):
# токен живет только в клиенте на время запроса, все данные - в памяти и уходят обратно в MWS Tables
def run_pipeline(token: str, space_id: str = None, spaces: list = None) -> dict:
    client = MWSTablesClient(token=token)

    # 1. Определяем пространство: переданное в запросе (с проверкой доступа) или ищем сами по папке Input
    log.info("1. Определение рабочего пространства")
    spaces = spaces if spaces is not None else client.get_spaces()

    if space_id:
        if space_id not in [s["id"] for s in spaces]:
            raise PipelineError(403, "Пространство недоступно для этого токена")
    else:
        try:
            space_id = client.find_space_id(INPUT_FOLDER, spaces)
        except LookupError as e:
            raise PipelineError(404, str(e))

    # 2. Ищем папку Input и получаем все таблицы внутри нее
    log.info("2. Поиск таблиц в папке '%s' (пространство %s)", INPUT_FOLDER, space_id)
    try:
        input_folder_id = client.get_folder(space_id, INPUT_FOLDER)
    except LookupError as e:
        raise PipelineError(404, str(e))

    input_tables = client.get_tables_in_folder(space_id, input_folder_id)
    if not input_tables:
        raise PipelineError(404, f"В папке '{INPUT_FOLDER}' нет таблиц")

    # 3. Скачиваем каждую таблицу в DataFrame
    log.info("3. Скачивание таблиц: %d", len(input_tables))
    dfs_by_name = {}

    for item in input_tables:
        name = item.get("name", item["id"])

        # Если в папке две таблицы с одинаковым названием, добавляем ID, чтобы одна не затерла другую
        if name in dfs_by_name:
            name = f"{name} ({item['id']})"

        dfs_by_name[name] = client.read_df(item["id"])

    # 4. Настройки пользователя + канонизация колонок, приведение валют и единиц к целевым, поиск дублей
    # и сведение в одну мастер-таблицу (Transform)
    log.info("4. Объединение таблиц (Transform)")
    config_df = load_config(client, space_id)
    master_df = build_master_table(dfs_by_name, config_df=config_df)
    if master_df.empty:
        raise PipelineError(422, f"Все таблицы в '{INPUT_FOLDER}' оказались без данных")

    # 5. Находим или создаем таблицу Output в корне и перезаписываем в ней данные
    log.info("5. Загрузка сводки в таблицу '%s'", OUTPUT_TABLE)
    output_table_id = client.get_or_create_table(space_id, master_df, OUTPUT_TABLE)

    try:
        uploaded = client.upload_df(output_table_id, master_df, clear=True)
    except ValueError:
        raise PipelineError(409, f"Колонки существующей таблицы '{OUTPUT_TABLE}' не совпадают со сводкой. "
                                 f"Удалите ее вручную - при следующем запуске она создастся заново")

    return {
        "space_id": space_id,
        "tables_processed": len(input_tables),
        "output_table_id": output_table_id,
        "rows_uploaded": uploaded
    }
