import hashlib
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config.get_config import config_data
from pgv import embedding as embedding_service
from pgv import write_db
from utils.logger import setup_logger
from utils.paths import PROJECT_ROOT

logger = setup_logger(__name__)


@dataclass
class SemanticMatch:
    """Structured semantic retrieval hit.

    Attributes:
        text (str): Retrieved document content.
        score (float): Distance score returned by vector search.
        metadata (dict[str, Any]): Metadata stored with the retrieved document.
    """

    text: str
    score: float
    metadata: Dict[str, Any]


class SchemaVectorService:
    """Manage schema indexing and semantic retrieval over PGVector."""

    def __init__(self, app_config: Dict[str, Any]) -> None:
        """Initialize service state from application configuration.

        Args:
            app_config (dict[str, Any]): Full application configuration dictionary.

        Returns:
            None: This constructor initializes internal fields.
        """
        vector_config = app_config.get("vector", {})
        self._vector_config: Dict[str, Any] = vector_config if isinstance(vector_config, dict) else {}
        self._enabled: bool = bool(self._vector_config.get("enabled", False))

        self._embedding_function: Optional[Any] = None
        self._vector_store: Optional[Any] = None
        self._initialized: bool = False
        self._schema_payload_hash: str = ""
        self._lock = Lock()

    def is_enabled(self) -> bool:
        """Check whether vector retrieval is enabled by configuration.

        Args:
            None.

        Returns:
            bool: ``True`` when vector retrieval is enabled.
        """
        return self._enabled

    def initialize(self) -> bool:
        """Initialize embedding model and vector store handles lazily.

        Args:
            None.

        Returns:
            bool: ``True`` when initialization succeeded.
        """
        if not self._enabled:
            return False

        with self._lock:
            if self._initialized and self._embedding_function is not None:
                return True

            try:
                self._embedding_function = embedding_service.get_embedding_function(
                    self._vector_config,
                    project_root=PROJECT_ROOT,
                )
                self._vector_store = write_db.get_store(self._vector_config, self._embedding_function)
                self._initialized = True
                logger.info("SchemaVectorService initialized")
                return True
            except Exception as exc:
                logger.warning("Failed to initialize SchemaVectorService: %s", exc)
                self._initialized = False
                self._vector_store = None
                return False

    def _normalize_schema_payload(
        self,
        schema_payload: Sequence[Any],
    ) -> Tuple[
        Dict[str, pd.DataFrame],
        Dict[str, Dict[str, str]],
        Dict[str, str],
        Dict[str, Dict[str, str]],
    ]:
        """Normalize schema payload from runtime data loader.

        Args:
            schema_payload (Sequence[Any]): Runtime schema payload in format
                ``[tables_data, foreign_keys, comments]``.

        Returns:
            tuple[dict[str, pandas.DataFrame], dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
            Normalized ``(tables_data, foreign_keys, table_comments, column_comments)``.
        """
        tables_data: Dict[str, pd.DataFrame] = {}
        foreign_keys: Dict[str, Dict[str, str]] = {}
        table_comments: Dict[str, str] = {}
        column_comments: Dict[str, Dict[str, str]] = {}

        if schema_payload and isinstance(schema_payload[0], dict):
            for table_name, table_df in schema_payload[0].items():
                if isinstance(table_df, pd.DataFrame):
                    tables_data[str(table_name)] = table_df

        if len(schema_payload) > 1 and isinstance(schema_payload[1], dict):
            for table_name, fk_mapping in schema_payload[1].items():
                if not isinstance(fk_mapping, dict):
                    continue
                foreign_keys[str(table_name)] = {
                    str(left): str(right)
                    for left, right in fk_mapping.items()
                }

        if (
            len(schema_payload) > 2
            and isinstance(schema_payload[2], tuple)
            and len(schema_payload[2]) == 2
        ):
            raw_table_comments, raw_column_comments = schema_payload[2]

            if isinstance(raw_table_comments, dict):
                table_comments = {
                    str(table_name): str(comment or "")
                    for table_name, comment in raw_table_comments.items()
                }

            if isinstance(raw_column_comments, dict):
                for table_name, column_mapping in raw_column_comments.items():
                    if not isinstance(column_mapping, dict):
                        continue
                    column_comments[str(table_name)] = {
                        str(column_name): str(comment or "")
                        for column_name, comment in column_mapping.items()
                    }

        return tables_data, foreign_keys, table_comments, column_comments

    def _build_schema_documents(
        self,
        schema_payload: Sequence[Any],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Convert database schema metadata into semantic documents.

        Args:
            schema_payload (Sequence[Any]): Runtime schema payload in format
                ``[tables_data, foreign_keys, comments]``.

        Returns:
            tuple[list[str], list[dict[str, Any]]]: Document texts and aligned metadata.
        """
        tables_data, foreign_keys, table_comments, column_comments = self._normalize_schema_payload(
            schema_payload,
        )

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for table_name in sorted(tables_data.keys()):
            table_df = tables_data[table_name]
            columns = [str(column) for column in table_df.columns.tolist()]

            commented_columns: List[str] = []
            table_column_comments = column_comments.get(table_name, {})
            for column_name in columns:
                column_comment = str(table_column_comments.get(column_name, "")).strip()
                if column_comment:
                    commented_columns.append(f"{column_name} ({column_comment})")
                else:
                    commented_columns.append(column_name)

            relation_mapping = foreign_keys.get(table_name, {})
            relation_text = "; ".join(
                f"{left_key}->{right_key}"
                for left_key, right_key in sorted(relation_mapping.items())
            )

            table_parts = [
                "Table schema reference.",
                f"Table name: {table_name}.",
                f"Columns: {', '.join(commented_columns)}." if commented_columns else "Columns: none.",
            ]

            table_comment = str(table_comments.get(table_name, "")).strip()
            if table_comment:
                table_parts.append(f"Table comment: {table_comment}.")

            if relation_text:
                table_parts.append(f"Foreign key relations: {relation_text}.")

            texts.append(" ".join(table_parts))
            metadatas.append({"doc_type": "table_schema", "table": table_name})

            for left_key, right_key in sorted(relation_mapping.items()):
                texts.append(f"Join relation: {left_key} references {right_key}.")
                metadatas.append(
                    {
                        "doc_type": "foreign_key",
                        "table": table_name,
                        "source": left_key,
                        "target": right_key,
                    }
                )

        return texts, metadatas

    def _calculate_payload_hash(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> str:
        """Calculate a stable hash for schema semantic documents.

        Args:
            texts (Sequence[str]): Semantic document texts.
            metadatas (Sequence[dict[str, Any]]): Metadata list aligned with ``texts``.

        Returns:
            str: SHA256 hash used to detect schema changes.
        """
        digest_source = json.dumps(
            {
                "texts": [str(item) for item in texts],
                "metadatas": [dict(item) for item in metadatas],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()

    def sync_schema_payload(
        self,
        schema_payload: Sequence[Any],
        force_rebuild: bool = False,
    ) -> int:
        """Synchronize schema metadata into PGVector documents.

        Args:
            schema_payload (Sequence[Any]): Runtime schema payload in format
                ``[tables_data, foreign_keys, comments]``.
            force_rebuild (bool, optional): Whether to force full rebuild even when
                schema hash is unchanged.

        Returns:
            int: Number of indexed semantic documents.
        """
        if not self.initialize() or self._embedding_function is None:
            return 0

        texts, metadatas = self._build_schema_documents(schema_payload)
        if not texts:
            return 0

        payload_hash = self._calculate_payload_hash(texts, metadatas)

        with self._lock:
            if (
                not force_rebuild
                and self._schema_payload_hash == payload_hash
                and self._vector_store is not None
            ):
                return 0

            try:
                self._vector_store = write_db.rebuild_collection(
                    vector_config=self._vector_config,
                    embedding_function=self._embedding_function,
                    texts=texts,
                    metadatas=metadatas,
                )
                self._schema_payload_hash = payload_hash
                logger.info("Synced %d schema semantic documents", len(texts))
                return len(texts)
            except Exception as exc:
                logger.warning("Failed to sync schema semantic documents: %s", exc)
                return 0

    def _resolve_top_k(self, custom_limit: Optional[int] = None) -> int:
        """Resolve top-k retrieval size from config and optional override.

        Args:
            custom_limit (Optional[int], optional): Optional explicit limit.

        Returns:
            int: Effective top-k value for retrieval.
        """
        if custom_limit is not None:
            try:
                return max(1, int(custom_limit))
            except (TypeError, ValueError):
                pass

        try:
            return max(1, int(self._vector_config.get("top_k", 6)))
        except (TypeError, ValueError):
            return 6

    def _resolve_max_distance(self) -> Optional[float]:
        """Resolve optional max distance threshold for search results.

        Args:
            None.

        Returns:
            Optional[float]: Maximum accepted distance, or ``None`` when disabled.
        """
        raw_value = self._vector_config.get("max_distance", None)
        if raw_value in (None, "", "none", "null"):
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    def retrieve(
        self,
        question: str,
        limit: Optional[int] = None,
    ) -> List[SemanticMatch]:
        """Retrieve schema hints semantically related to a question.

        Args:
            question (str): User natural-language question.
            limit (Optional[int], optional): Maximum number of hits.

        Returns:
            list[SemanticMatch]: Filtered semantic matches.
        """
        normalized_question = str(question).strip()
        if not normalized_question:
            return []
        if not self.initialize() or self._vector_store is None:
            return []

        max_distance = self._resolve_max_distance()
        top_k = self._resolve_top_k(limit)

        try:
            raw_matches = write_db.similarity_search_with_score(
                store=self._vector_store,
                query=normalized_question,
                limit=top_k,
            )
        except Exception as exc:
            logger.warning("Vector similarity search failed: %s", exc)
            return []

        matches: List[SemanticMatch] = []
        for document, score in raw_matches:
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue

            if max_distance is not None and numeric_score > max_distance:
                continue

            page_content = str(getattr(document, "page_content", "")).strip()
            if not page_content:
                continue

            metadata = getattr(document, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            matches.append(
                SemanticMatch(
                    text=page_content,
                    score=numeric_score,
                    metadata=dict(metadata),
                )
            )
        return matches

    def build_prompt_context(self, question: str) -> str:
        """Build prompt context text from semantic retrieval hits.

        Args:
            question (str): User natural-language question.

        Returns:
            str: A formatted prompt block, or empty string when no hits are available.
        """
        matches = self.retrieve(question)
        if not matches:
            return ""

        lines: List[str] = [
            "Semantic schema hints retrieved from the vector database:",
        ]
        for index, item in enumerate(matches, start=1):
            table_name = str(item.metadata.get("table", "")).strip()
            table_prefix = f"[table={table_name}] " if table_name else ""
            compact_text = " ".join(item.text.split())
            lines.append(
                f"{index}. distance={item.score:.4f}; {table_prefix}{compact_text}",
            )

        lines.append(
            "Use these hints as reliable references for table names, column names, and join paths.",
        )
        return "\n".join(lines)


_SERVICE_LOCK = Lock()
_SERVICE_INSTANCE: Optional[SchemaVectorService] = None


def get_schema_vector_service() -> SchemaVectorService:
    """Return the singleton schema vector service.

    Args:
        None.

    Returns:
        SchemaVectorService: Shared service instance for semantic retrieval.
    """
    global _SERVICE_INSTANCE
    with _SERVICE_LOCK:
        if _SERVICE_INSTANCE is None:
            _SERVICE_INSTANCE = SchemaVectorService(config_data)
        return _SERVICE_INSTANCE


def sync_schema_knowledge(schema_payload: Sequence[Any], force_rebuild: bool = False) -> int:
    """Synchronize runtime schema payload into vector database knowledge.

    Args:
        schema_payload (Sequence[Any]): Runtime schema payload in format
            ``[tables_data, foreign_keys, comments]``.
        force_rebuild (bool, optional): Whether to force full rebuild of collection.

    Returns:
        int: Number of indexed documents.
    """
    service = get_schema_vector_service()
    return service.sync_schema_payload(schema_payload, force_rebuild=force_rebuild)


def build_semantic_context(question: str) -> str:
    """Build vector-retrieved semantic context for prompt assembly.

    Args:
        question (str): User natural-language question.

    Returns:
        str: Prompt context block derived from vector search, or empty string.
    """
    service = get_schema_vector_service()
    return service.build_prompt_context(question)
