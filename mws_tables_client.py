import os
import math
import time
import requests
import pandas as pd

BASE_URL = os.getenv("MWS_BASE_URL", "https://tables.mws.ru/fusion/v1")
PAGE_SIZE = 1000  # максимум записей на страницу при чтении
BATCH_SIZE = 10  # лимит API: не более 10 записей за один POST / DELETE
REQUEST_DELAY = 0.2  # пауза между запросами: не более 5 запросов в секунду
MAX_RETRIES = 5  # сколько раз повторяем запрос при статусе 429 или обрыве соединения
TIMEOUT = 30


class MWSTablesClient:
    def __init__(self, token: str, base_url: str = BASE_URL):
        self.__token = token
        self.__base_url = base_url.rstrip("/")
        self.__headers = {
            "Authorization": f"Bearer {token}"
        }

    # Единая точка отправки запросов: повторяет запрос при 429 (Rate Limit) и обрыве соединения,
    # кидает исключение на HTTP-ошибки и на ответы вида {"success": false} (API отдает их с HTTP 200)
    def __request(self, method: str, path: str, params: dict = None, json: dict = None) -> dict:
        url = f"{self.__base_url}{path}"

        # Content-Type добавляем только если есть тело: с пустым телом API отвечает 500
        headers = {**self.__headers, "Content-Type": "application/json"} if json is not None else self.__headers

        for attempt in range(MAX_RETRIES):
            try:
                #time.sleep(REQUEST_DELAY)
                resp = requests.request(method, url, params=params, json=json, headers=headers, timeout=TIMEOUT)
            except requests.ConnectionError:
                # POST не повторяем: запрос мог дойти до сервера, и записи задублируются
                if method == "POST" or attempt == MAX_RETRIES - 1:
                    raise

                time.sleep(REQUEST_DELAY * 2 ** (attempt + 1))
                continue

            if resp.status_code == 429:
                # ждем время из Retry-After (если API его отдал), иначе растим паузу экспоненциально
                delay = float(resp.headers.get("Retry-After", REQUEST_DELAY * 2 ** (attempt + 1)))
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data = resp.json()

            if data.get("success") is False:
                raise RuntimeError(f"{method} {path}: {data.get('message')} (code {data.get('code')})")

            return data

        raise RuntimeError(f"Превышен лимит запросов (429) для {method} {path}")

    def get(self, path: str, params: dict = None) -> dict:
        return self.__request("GET", path, params=params)

    def post(self, path: str, json: dict) -> dict:
        return self.__request("POST", path, json=json)

    def delete(self, path: str, params: dict = None) -> dict:
        return self.__request("DELETE", path, params=params)

    # Список рабочих пространств токена: [{"id": "spc...", "name": "..."}, ...]
    def get_spaces(self) -> list:
        return self.get("/spaces").get("data", {}).get("spaces", [])

    # Автопоиск пространства: первое из spaces (по умолчанию - все пространства токена), где в корне есть папка folder_name
    def find_space_id(self, folder_name: str = "Input", spaces: list = None) -> str:
        for space in spaces if spaces is not None else self.get_spaces():
            try:
                self.get_folder(space["id"], folder_name)
                return space["id"]
            except LookupError:
                continue

        raise LookupError(f"Не найдено пространство с папкой '{folder_name}'")

    # Узлы (папки и таблицы) в корне пространства или внутри папки parent_id.
    # Параметр ?parentId в листинге API игнорирует (всегда отдает корень), поэтому содержимое папки
    # берем из детальной информации об узле: GET /spaces/{space_id}/nodes/{parent_id} -> data.children
    def __get_nodes(self, space_id: str, parent_id: str = None) -> list:
        if parent_id:
            return self.get(f"/spaces/{space_id}/nodes/{parent_id}").get("data", {}).get("children", [])

        return self.get(f"/spaces/{space_id}/nodes").get("data", {}).get("nodes", [])

    # Ищет папку по имени (без учета регистра) и возвращает ее id.
    # API не умеет создавать папки, поэтому если папки нет - кидаем понятную ошибку
    def get_folder(self, space_id: str, folder_name: str = "Input", parent_id: str = None) -> str:
        for node in self.__get_nodes(space_id, parent_id):
            if node.get("type") == "Folder" and node.get("name", "").lower() == folder_name.lower():
                return node["id"]

        raise LookupError(f"Папка '{folder_name}' не найдена в пространстве {space_id}. Создайте ее вручную")

    # Все таблицы (type == "Datasheet") внутри папки: [{"id": "dst...", "name": "..."}, ...]
    def get_tables_in_folder(self, space_id: str, folder_id: str) -> list:
        return [node for node in self.__get_nodes(space_id, folder_id) if node.get("type") == "Datasheet"]

    # Ищет таблицу по имени (без учета регистра) в корне пространства или в папке folder_id.
    # title - одно имя или несколько допустимых (берется первая найденная). Возвращает id или None, если таблицы нет
    def find_table(self, space_id: str, title, folder_id: str = None) -> str:
        titles = {t.lower() for t in ([title] if isinstance(title, str) else title)}

        for node in self.__get_nodes(space_id, folder_id):
            if node.get("type") == "Datasheet" and node.get("name", "").lower() in titles:
                return node["id"]

        return None

    # Описание колонок таблицы (имя, тип и т.д.) строго в том порядке, как они идут в интерфейсе (слева направо)
    def get_fields(self, datasheet_id: str) -> list:
        return self.get(f"/datasheets/{datasheet_id}/fields").get("data", {}).get("fields", [])

    # Названия колонок таблицы в порядке интерфейса
    def get_cols(self, datasheet_id: str) -> list:
        return [f["name"] for f in self.get_fields(datasheet_id)]

    # Читает всю таблицу (или конкретное представление view_id) в DataFrame
    def read_df(self, datasheet_id: str, view_id: str = None) -> pd.DataFrame:
        ordered_cols = self.get_cols(datasheet_id)
        rows = []
        page_num = 1

        # Пагинация: идем по страницам, пока API не вернет неполную (последнюю) страницу
        while True:
            params = {"pageNum": page_num, "pageSize": PAGE_SIZE}
            if view_id:
                params["viewId"] = view_id

            records = self.get(f"/datasheets/{datasheet_id}/records", params=params).get("data", {}).get("records", [])

            # Пустые строки-заготовки приходят с "fields": {} - отбрасываем их
            rows.extend(r["fields"] for r in records if r.get("fields"))

            if len(records) < PAGE_SIZE:
                break

            page_num += 1
            time.sleep(REQUEST_DELAY)

        # reindex фиксирует порядок колонок как в интерфейсе и добавляет полностью пустые колонки,
        # которых нет ни в одной записи (иначе они бы пропали из DataFrame)
        return pd.DataFrame(rows).reindex(columns=ordered_cols)

    # Определяет тип колонки MWS по dtype pandas: bool -> Checkbox, числа -> Number, остальное -> SingleText.
    # У каждого типа API требует property; первая колонка таблицы (primary) не может быть Checkbox
    def __get_field_type(self, series: pd.Series, is_first: bool) -> dict:
        if pd.api.types.is_bool_dtype(series) and not is_first:
            return {"type": "Checkbox", "property": {"icon": "white_check_mark"}}

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return {"type": "Number", "property": {"precision": 2}}

        return {"type": "SingleText", "property": {}}

    # Удаляет все записи таблицы (по 10 штук за запрос). Нужно, чтобы убрать пустые строки-заготовки
    # новой таблицы и чтобы повторный запуск пайплайна не дублировал данные в Output
    def clear_table(self, datasheet_id: str) -> int:
        deleted = 0

        while True:
            records = self.get(f"/datasheets/{datasheet_id}/records", params={"pageSize": PAGE_SIZE}) \
                .get("data", {}).get("records", [])
            record_ids = [r["recordId"] for r in records]

            if not record_ids:
                return deleted

            # Каждый раз читаем первую страницу заново: удаленные записи из нее пропадают
            for i in range(0, len(record_ids), BATCH_SIZE):
                self.delete(f"/datasheets/{datasheet_id}/records", params={"recordIds": record_ids[i:i + BATCH_SIZE]})
                time.sleep(REQUEST_DELAY)

            deleted += len(record_ids)

    # Создает таблицу с колонками из df (в том же порядке) и возвращает ее id (dst...)
    def create_table_from_df(self, space_id: str, title: str, df: pd.DataFrame, folder_id: str = None) -> str:
        fields = [{"name": str(col), **self.__get_field_type(df[col], i == 0)} for i, col in enumerate(df.columns)]

        body = {"name": title, "fields": fields}
        if folder_id:
            body["folderId"] = folder_id

        datasheet_id = self.post(f"/spaces/{space_id}/datasheets", body)["data"]["id"]

        # MWS создает таблицу с 3 пустыми строками - убираем их, чтобы данные начинались с первой строки
        self.clear_table(datasheet_id)

        return datasheet_id

    # Находит таблицу title (по умолчанию "Output") в корне пространства, если ее нет - создает по колонкам df.
    # Это единственный объект, который API умеет создавать
    def get_or_create_table(self, space_id: str, df: pd.DataFrame, title: str = "Output", folder_id: str = None) -> str:
        return self.find_table(space_id, title, folder_id) or self.create_table_from_df(space_id, title, df, folder_id)

    # Приводит значение ячейки к виду, который можно отправить в JSON; пустое значение -> None
    def __clean_value(self, value):
        if isinstance(value, (list, dict)):
            return value

        # NaN / NaT / None / пустая строка считаем пустой ячейкой
        if pd.isna(value) or value == "":
            return None

        # numpy-типы (int64, float64, bool_) не сериализуются в JSON - переводим в обычные python-типы
        if hasattr(value, "item"):
            value = value.item()

        # inf тоже не валиден для JSON
        if isinstance(value, float) and not math.isfinite(value):
            return None

        # даты и прочие объекты отправляем строкой
        if not isinstance(value, (str, int, float, bool)):
            return str(value)

        return value

    # Приводит значение к типу колонки таблицы: API не принимает, например, число в текстовую колонку
    def __cast_value(self, value, field_type: str):
        if value is None:
            return None

        if field_type in ("SingleText", "Text"):
            # 3.0 -> "3", чтобы в тексте не появлялись лишние ".0"
            if isinstance(value, float) and value.is_integer():
                return str(int(value))

            return str(value)

        if field_type == "Number":
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if field_type == "Checkbox":
            return bool(value)

        return value

    # Заливает строки df в таблицу пачками по 10 записей; возвращает количество загруженных строк
    def upload_df(self, datasheet_id: str, df: pd.DataFrame, clear: bool = False) -> int:
        # Колонки добавлять через API нельзя, поэтому все колонки df должны уже быть в таблице
        field_types = {f["name"]: f.get("type") for f in self.get_fields(datasheet_id)}
        missing_cols = [str(col) for col in df.columns if str(col) not in field_types]
        if missing_cols:
            raise ValueError(f"В таблице {datasheet_id} нет колонок: {missing_cols}")

        # Повторный запуск: очищаем таблицу, чтобы не дублировать старые данные
        if clear:
            self.clear_table(datasheet_id)

        records = []

        for row in df.to_dict(orient="records"):
            # пустые значения не передаем в fields вообще
            fields = {}
            for col, val in row.items():
                val = self.__cast_value(self.__clean_value(val), field_types[str(col)])
                if val is not None:
                    fields[str(col)] = val

            # строку без единого значения пропускаем
            if fields:
                records.append({"fields": fields})

        # Батчи: API принимает не более 10 записей за запрос
        for i in range(0, len(records), BATCH_SIZE):
            self.post(f"/datasheets/{datasheet_id}/records", {"records": records[i:i + BATCH_SIZE]})
            time.sleep(REQUEST_DELAY)  # не превышаем 5 запросов в секунду

        return len(records)
