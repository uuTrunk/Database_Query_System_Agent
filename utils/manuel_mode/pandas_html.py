import pygwalker as pyg
from pandas import DataFrame

from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_html(df: DataFrame) -> str:
    """Convert a DataFrame to an interactive PyGWalker HTML string.

    Args:
        df (pandas.DataFrame): Source dataframe to visualize.

    Returns:
        str: Generated HTML content, or an empty string if conversion fails.
    """
    try:
        html_str = pyg.to_html(df, dark='light')
    except Exception as e:
        logger.exception("Failed to render DataFrame to PyGWalker HTML: %s", e)
        html_str = ""

    return html_str
