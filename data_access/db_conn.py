from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.get_config import config_data
from utils.logger import setup_logger

logger = setup_logger(__name__)

engine = create_engine(
	config_data["mysql"],
	pool_pre_ping=True,
	pool_recycle=1800,
)


def verify_database_connection() -> None:
	"""Verify database connectivity at startup.

	Args:
		None.

	Returns:
		None: This function succeeds silently when the database is reachable.

	Raises:
		RuntimeError: If the database test query fails.
	"""
	try:
		with engine.connect() as connection:
			connection.execute(text("SELECT 1"))
		logger.info("Database connection verified")
	except SQLAlchemyError as exc:
		raise RuntimeError(f"Unable to connect to database: {exc}") from exc


verify_database_connection()

