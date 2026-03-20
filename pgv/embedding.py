from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import setup_logger

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - compatibility fallback
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore[import-not-found]

logger = setup_logger(__name__)

_EMBEDDING_CACHE: Dict[Tuple[str, str], HuggingFaceEmbeddings] = {}
_CACHE_LOCK = Lock()

DEFAULT_EMBEDDING_MODEL = "shibing624/text2vec-base-multilingual"
DEFAULT_EMBEDDING_DEVICE = "cpu"


def _normalize_device(vector_config: Dict[str, Any]) -> str:
    """Normalize embedding device configuration.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.

    Returns:
        str: Normalized device string such as ``cpu`` or ``cuda``.
    """
    raw_value = str(vector_config.get("embedding_device", DEFAULT_EMBEDDING_DEVICE)).strip()
    return raw_value or DEFAULT_EMBEDDING_DEVICE


def resolve_embedding_model_name(
    vector_config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> str:
    """Resolve an embedding model identifier from configuration.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        project_root (Optional[pathlib.Path], optional): Project root used to resolve
            relative local model paths.

    Returns:
        str: Hugging Face model id or absolute local model path.
    """
    raw_model_name = str(vector_config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)).strip()
    model_name = raw_model_name or DEFAULT_EMBEDDING_MODEL

    if project_root and model_name.startswith("."):
        return str((project_root / model_name).resolve())
    return model_name


def get_embedding_function(
    vector_config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> HuggingFaceEmbeddings:
    """Create or reuse a HuggingFace embedding function.

    Args:
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        project_root (Optional[pathlib.Path], optional): Project root used to resolve
            relative model paths.

    Returns:
        HuggingFaceEmbeddings: Cached embedding function instance.
    """
    model_name = resolve_embedding_model_name(vector_config, project_root=project_root)
    device = _normalize_device(vector_config)
    cache_key = (model_name, device)

    with _CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            embedding_function = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except TypeError:
            # Older LangChain versions do not support encode_kwargs.
            embedding_function = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
            )

        _EMBEDDING_CACHE[cache_key] = embedding_function
        logger.info("Loaded embedding model '%s' on device '%s'", model_name, device)
        return embedding_function


def embed_texts(
    texts: List[str],
    vector_config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> List[List[float]]:
    """Embed a list of texts using the configured embedding model.

    Args:
        texts (list[str]): Input texts to embed.
        vector_config (dict[str, Any]): Vector module configuration dictionary.
        project_root (Optional[pathlib.Path], optional): Project root used to resolve
            relative model paths.

    Returns:
        list[list[float]]: Dense vectors generated for each input text.
    """
    if not texts:
        return []
    embedding_function = get_embedding_function(vector_config, project_root=project_root)
    return embedding_function.embed_documents(texts)
