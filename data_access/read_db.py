from threading import Lock
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from data_access.db_conn import engine
from utils.logger import setup_logger

logger = setup_logger(__name__)

tables_data = None
foreign_keys_cache = None
comments_cache = None
tables_data_lock = Lock()


def _build_select_all_query(table_name: str, columns: List[str]):
    """Build a deterministic SELECT statement with explicit columns.

    Args:
        table_name: Physical table name from SQLAlchemy inspector.
        columns: Ordered table column names from SQLAlchemy inspector.

    Returns:
        sqlalchemy.sql.elements.TextClause: SQL text object.
    """
    quoted_columns = ", ".join(f"`{column}`" for column in columns)
    return text(f"SELECT {quoted_columns} FROM `{table_name}`")


def get_foreign_keys(inspector=None) -> Dict[str, Dict[str, str]]:
    """Load foreign-key relationships for all tables in the current database.

    Args:
        None.

    Returns:
        dict[str, dict[str, str]]: Mapping in format
        ``{table: {table.column: referred_table.referred_column}}``.

    Raises:
        RuntimeError: If database metadata introspection fails.
    """
    try:
        inspector = inspector or inspect(engine)
        foreign_keys: Dict[str, Dict[str, str]] = {}
        for table_name in inspector.get_table_names():
            fks = inspector.get_foreign_keys(table_name)
            if not fks:
                continue

            foreign_keys[table_name] = {}
            for fk in fks:
                constrained_columns = fk.get("constrained_columns", [])
                referred_columns = fk.get("referred_columns", [])
                referred_table = fk.get("referred_table")
                for index, constrained_column in enumerate(constrained_columns):
                    if index < len(referred_columns) and referred_table:
                        foreign_keys[table_name][
                            f"{table_name}.{constrained_column}"
                        ] = f"{referred_table}.{referred_columns[index]}"
        return foreign_keys
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to load foreign keys: {exc}") from exc


def get_table_and_column_comments(inspector=None) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Load table comments and column comments for all tables.

    Args:
        None.

    Returns:
        tuple[dict[str, str], dict[str, dict[str, str]]]:
        ``(table_comments, column_comments)`` where:
        - ``table_comments`` maps table name to comment text.
        - ``column_comments`` maps table name to ``{column_name: comment}``.

    Raises:
        RuntimeError: If metadata loading fails.
    """
    try:
        inspector = inspector or inspect(engine)
        table_comments: Dict[str, str] = {}
        column_comments: Dict[str, Dict[str, str]] = {}
        for table_name in inspector.get_table_names():
            table_comment_info = inspector.get_table_comment(table_name) or {}
            table_comments[table_name] = table_comment_info.get("text") or ""

            column_comments[table_name] = {}
            for column in inspector.get_columns(table_name):
                column_name = str(column.get("name", ""))
                column_comments[table_name][column_name] = str(column.get("comment") or "")

        return table_comments, column_comments
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to load table comments: {exc}") from exc


def _load_tables_data(inspector=None) -> Dict[str, pd.DataFrame]:
    """Read all database tables into pandas DataFrames.

    Args:
        None.

    Returns:
        dict[str, pandas.DataFrame]: Mapping from table name to full table DataFrame.

    Raises:
        RuntimeError: If table listing or data querying fails.
    """
    try:
        inspector = inspector or inspect(engine)
        loaded_tables: Dict[str, pd.DataFrame] = {}
        with engine.connect() as connection:
            for table_name in inspector.get_table_names():
                columns = [str(column.get("name")) for column in inspector.get_columns(table_name)]
                query = _build_select_all_query(table_name, columns)
                loaded_tables[table_name] = pd.read_sql(query, connection)
        logger.info("Loaded %d tables from database", len(loaded_tables))
        return loaded_tables
    except Exception as exc:
        raise RuntimeError(f"Failed to load table data from database: {exc}") from exc


def get_data_from_db(force_reload: bool = False):
    """Get cached table data plus schema metadata for prompt construction.

    Args:
        force_reload (bool, optional): Whether to bypass cache and reload all table data.

    Returns:
        tuple[dict[str, pandas.DataFrame], dict[str, dict[str, str]], tuple[dict[str, str], dict[str, dict[str, str]]]]:
        A tuple ``(tables_data, foreign_keys, comments)`` where:
        - ``tables_data`` maps table names to DataFrames.
        - ``foreign_keys`` describes table relationships.
        - ``comments`` is ``(table_comments, column_comments)``.
    """
    global tables_data
    global foreign_keys_cache
    global comments_cache

    if force_reload or tables_data is None or foreign_keys_cache is None or comments_cache is None:
        with tables_data_lock:
            if force_reload or tables_data is None or foreign_keys_cache is None or comments_cache is None:
                inspector = inspect(engine)
                tables_data = _load_tables_data(inspector=inspector)
                foreign_keys_cache = get_foreign_keys(inspector=inspector)
                comments_cache = get_table_and_column_comments(inspector=inspector)

    return tables_data, foreign_keys_cache, comments_cache

