from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logger import setup_logger

try:
    from langchain_community.vectorstores import PGVector
    from langchain_community.vectorstores.pgvector import DistanceStrategy
except Exception:  # pragma: no cover - compatibility fallback
    from langchain.vectorstores.pgvector import DistanceStrategy, PGVector

logger = setup_logger(__name__)

DEFAULT_COLLECTION_NAME = "schema_knowledge"
DEFAULT_DISTANCE_STRATEGY = "cosine"


def _to_int(value: Any, default: int) -> int:
    """Convert a value to integer with fallback.

    Args:
        value (Any): Value to parse.
        default (int): Fallback integer when parsing fails.

    Returns:
        int: Parsed integer value or ``default``.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_distance_strategy(raw_value: Any) -> DistanceStrategy:
    """Resolve distance strategy from textual configuration.

    Args:
        raw_value (Any): Text value such as ``cosine`` or ``euclidean``.

    Returns:
        DistanceStrategy: PGVector-compatible distance strategy enum.
    """
    normalized = str(raw_value or DEFAULT_DISTANCE_STRATEGY).strip().lower()
    mapping = {
        "cosine": DistanceStrategy.COSINE,
        "euclidean": DistanceStrategy.EUCLIDEAN,
        "l2": DistanceStrategy.EUCLIDEAN,
        "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
        "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
    }
    return mapping.get(normalized, DistanceStrategy.COSINE)


def build_connection_string(vector_config: Dict[str, Any]) -> str:
    """Build a PGVector SQLAlchemy connection string from configuration.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.

    Returns:
        str: SQLAlchemy-style PostgreSQL connection string.
    """
    direct = str(vector_config.get("connection_string", "")).strip()
    if direct:
        return direct

    db_config = vector_config.get("db", {})
    if not isinstance(db_config, dict):
        db_config = {}

    return PGVector.connection_string_from_db_params(
        driver=str(db_config.get("driver", "psycopg2")),
        host=str(db_config.get("host", "127.0.0.1")),
        port=_to_int(db_config.get("port", 5434), 5434),
        database=str(db_config.get("database", "test")),
        user=str(db_config.get("user", "postgres")),
        password=str(db_config.get("password", "123456")),
    )


def _create_store(
    vector_config: Dict[str, Any],
    embedding_function: Any,
    pre_delete_collection: bool = False,
) -> PGVector:
    """Create a PGVector store instance.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        embedding_function (Any): Embedding function object accepted by PGVector.
        pre_delete_collection (bool, optional): Whether to clear the existing
            collection before use.

    Returns:
        PGVector: Connected vector store instance.
    """
    kwargs = {
        "connection_string": build_connection_string(vector_config),
        "embedding_function": embedding_function,
        "collection_name": str(vector_config.get("collection_name", DEFAULT_COLLECTION_NAME)),
        "distance_strategy": _resolve_distance_strategy(
            vector_config.get("distance_strategy", DEFAULT_DISTANCE_STRATEGY),
        ),
        "pre_delete_collection": bool(pre_delete_collection),
    }

    try:
        return PGVector(**kwargs)
    except TypeError:
        # Compatibility for older constructors that do not accept pre_delete_collection.
        kwargs.pop("pre_delete_collection", None)
        return PGVector(**kwargs)


def get_store(vector_config: Dict[str, Any], embedding_function: Any) -> PGVector:
    """Get a PGVector store handle without resetting collection contents.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        embedding_function (Any): Embedding function object accepted by PGVector.

    Returns:
        PGVector: Connected vector store instance.
    """
    return _create_store(
        vector_config=vector_config,
        embedding_function=embedding_function,
        pre_delete_collection=False,
    )


def rebuild_collection(
    vector_config: Dict[str, Any],
    embedding_function: Any,
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> PGVector:
    """Rebuild a vector collection from a full list of texts.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        embedding_function (Any): Embedding function object accepted by PGVector.
        texts (Sequence[str]): Full text list to insert into the collection.
        metadatas (Optional[Sequence[dict[str, Any]]], optional): Metadata list
            aligned by index with ``texts``.

    Returns:
        PGVector: Rebuilt vector store containing inserted documents.

    Raises:
        ValueError: If ``texts`` is empty or metadata length mismatches text length.
    """
    if not texts:
        raise ValueError("Cannot rebuild vector collection with an empty text list.")

    metadata_list: Optional[List[Dict[str, Any]]] = None
    if metadatas is not None:
        metadata_list = [dict(item) for item in metadatas]
        if len(metadata_list) != len(texts):
            raise ValueError("Metadata count must match text count when provided.")

    store = _create_store(
        vector_config=vector_config,
        embedding_function=embedding_function,
        pre_delete_collection=True,
    )

    store.add_texts(
        texts=[str(item) for item in texts],
        metadatas=metadata_list,
    )
    logger.info(
        "Rebuilt vector collection '%s' with %d documents",
        vector_config.get("collection_name", DEFAULT_COLLECTION_NAME),
        len(texts),
    )
    return store


def similarity_search_with_score(
    store: PGVector,
    query: str,
    limit: int,
) -> List[Tuple[Any, float]]:
    """Run similarity search with score from an existing PGVector store.

    Args:
        store (PGVector): Vector store object.
        query (str): User query text.
        limit (int): Maximum number of matches.

    Returns:
        list[tuple[Any, float]]: List of ``(document, score)`` matches.
    """
    normalized_query = str(query).strip()
    if not normalized_query:
        return []
    top_k = max(1, _to_int(limit, 1))
    return store.similarity_search_with_score(normalized_query, k=top_k)
