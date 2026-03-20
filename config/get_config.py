from pathlib import Path
from typing import Any, Dict

import yaml

from utils.paths import CONFIG_FILE, DEFAULT_SERVER_PORT

REQUIRED_TOP_LEVEL_KEYS = ("mysql", "llm", "ai")
REQUIRED_LLM_KEYS = ("model_provider", "model")
REQUIRED_AI_KEYS = ("tries", "wait", "data_rows")

VECTOR_DEFAULTS = {
    "enabled": False,
    "embedding_model": "shibing624/text2vec-base-multilingual",
    "embedding_device": "cpu",
    "collection_name": "schema_knowledge",
    "top_k": 6,
    "max_distance": None,
    "distance_strategy": "cosine",
    "connection_string": "",
    "db": {
        "driver": "psycopg2",
        "host": "127.0.0.1",
        "port": 5434,
        "database": "test",
        "user": "postgres",
        "password": "123456",
    },
}


def _to_bool(value: Any, default: bool = False) -> bool:
    """Convert arbitrary values to boolean with readable coercion rules.

    Args:
        value (Any): Value to convert.
        default (bool, optional): Fallback value for unsupported string tokens.

    Returns:
        bool: Converted boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _normalize_vector_config(raw_value: Any) -> Dict[str, Any]:
    """Normalize optional vector module configuration.

    Args:
        raw_value (Any): Raw ``vector`` section from YAML.

    Returns:
        dict[str, Any]: Normalized vector configuration with default values.

    Raises:
        TypeError: If ``vector`` or ``vector.db`` has invalid type.
        ValueError: If numeric fields cannot be parsed.
    """
    if raw_value is None:
        vector_raw: Dict[str, Any] = {}
    elif isinstance(raw_value, dict):
        vector_raw = dict(raw_value)
    else:
        raise TypeError("`vector` section must be a dictionary when provided.")

    raw_db = vector_raw.get("db", {})
    if raw_db is None:
        raw_db = {}
    if not isinstance(raw_db, dict):
        raise TypeError("`vector.db` section must be a dictionary when provided.")

    db_defaults = VECTOR_DEFAULTS["db"]
    normalized_db = {
        "driver": str(raw_db.get("driver", db_defaults["driver"])).strip() or db_defaults["driver"],
        "host": str(raw_db.get("host", db_defaults["host"])).strip() or db_defaults["host"],
        "port": int(raw_db.get("port", db_defaults["port"])),
        "database": str(raw_db.get("database", db_defaults["database"])).strip() or db_defaults["database"],
        "user": str(raw_db.get("user", db_defaults["user"])).strip() or db_defaults["user"],
        "password": str(raw_db.get("password", db_defaults["password"])),
    }

    raw_max_distance = vector_raw.get("max_distance", VECTOR_DEFAULTS["max_distance"])
    if raw_max_distance in (None, "", "none", "null"):
        normalized_max_distance = None
    else:
        normalized_max_distance = float(raw_max_distance)

    normalized_top_k = int(vector_raw.get("top_k", VECTOR_DEFAULTS["top_k"]))
    normalized_top_k = max(1, normalized_top_k)

    normalized = {
        "enabled": _to_bool(vector_raw.get("enabled", VECTOR_DEFAULTS["enabled"]), default=False),
        "embedding_model": str(
            vector_raw.get("embedding_model", VECTOR_DEFAULTS["embedding_model"]),
        ).strip()
        or VECTOR_DEFAULTS["embedding_model"],
        "embedding_device": str(
            vector_raw.get("embedding_device", VECTOR_DEFAULTS["embedding_device"]),
        ).strip()
        or VECTOR_DEFAULTS["embedding_device"],
        "collection_name": str(
            vector_raw.get("collection_name", VECTOR_DEFAULTS["collection_name"]),
        ).strip()
        or VECTOR_DEFAULTS["collection_name"],
        "top_k": normalized_top_k,
        "max_distance": normalized_max_distance,
        "distance_strategy": str(
            vector_raw.get("distance_strategy", VECTOR_DEFAULTS["distance_strategy"]),
        ).strip().lower()
        or VECTOR_DEFAULTS["distance_strategy"],
        "connection_string": str(
            vector_raw.get("connection_string", VECTOR_DEFAULTS["connection_string"]),
        ).strip(),
        "db": normalized_db,
    }
    return normalized


def _validate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required configuration keys and normalize optional fields.

    Args:
        data (dict[str, Any]): Parsed YAML configuration dictionary.

    Returns:
        dict[str, Any]: Validated and normalized configuration dictionary.

    Raises:
        KeyError: If required top-level or nested keys are missing.
        TypeError: If nested config sections are not dictionaries.
        ValueError: If numeric values cannot be converted to expected numeric types.
    """
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing required top-level config keys: {missing}")

    if not isinstance(data["llm"], dict):
        raise TypeError("`llm` section must be a dictionary.")
    if not isinstance(data["ai"], dict):
        raise TypeError("`ai` section must be a dictionary.")

    llm_missing = [key for key in REQUIRED_LLM_KEYS if key not in data["llm"]]
    if llm_missing:
        raise KeyError(f"Missing required llm config keys: {llm_missing}")

    ai_missing = [key for key in REQUIRED_AI_KEYS if key not in data["ai"]]
    if ai_missing:
        raise KeyError(f"Missing required ai config keys: {ai_missing}")

    data["ai"]["tries"] = int(data["ai"]["tries"])
    data["ai"]["wait"] = int(data["ai"]["wait"])
    data["ai"]["data_rows"] = int(data["ai"]["data_rows"])

    if "server" in data and isinstance(data["server"], dict) and "port" in data["server"]:
        data["server_port"] = int(data["server"]["port"])
    else:
        data["server_port"] = int(data.get("server_port", DEFAULT_SERVER_PORT))

    data["vector"] = _normalize_vector_config(data.get("vector", {}))

    return data


def load_config(config_file: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load and validate configuration from a YAML file.

    Args:
        config_file (Path, optional): Absolute or relative path to the YAML config file.

    Returns:
        dict[str, Any]: Parsed and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the provided config file path does not exist.
        ValueError: If the YAML is invalid.
        TypeError: If YAML root object is not a dictionary.
        KeyError: If required keys are missing.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {config_path}") from exc

    if not isinstance(loaded, dict):
        raise TypeError(f"Config file must contain a YAML object: {config_path}")

    return _validate_config(loaded)


try:
    config_data = load_config()
except Exception as exc:
    raise RuntimeError(f"Failed to load configuration from {CONFIG_FILE}: {exc}") from exc



