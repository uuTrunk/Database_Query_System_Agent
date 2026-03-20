import re
from typing import Any, Optional

import pandas

_PNG_PATTERN = re.compile(r"tmp_imgs/[^\s/]+\.png")
_HTML_PATTERN = re.compile(r"tmp_imgs/[^\s/]+\.html")
_PYTHON_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_BLOCK_PATTERN = re.compile(r"```\s*(.*?)```", re.DOTALL)


def _first_match(pattern: re.Pattern[str], txt: Any) -> Optional[str]:
    """Return the first regex match from arbitrary input.

    Args:
        pattern (re.Pattern[str]): Compiled regex pattern used for searching.
        txt (Any): Source object converted to string before matching.

    Returns:
        Optional[str]: First match string, or ``None`` when no match exists.
    """
    matches = pattern.findall(str(txt))
    return matches[0] if matches else None


def parse_output_img(txt: Any) -> Optional[str]:
    """Extract the first PNG path under ``tmp_imgs/`` from model output text.

    Args:
        txt (Any): Raw model output to parse.

    Returns:
        Optional[str]: Matched PNG relative path, or ``None`` when not found.
    """
    return _first_match(_PNG_PATTERN, txt)


def parse_output_html(txt: Any) -> Optional[str]:
    """Extract the first HTML path under ``tmp_imgs/`` from model output text.

    Args:
        txt (Any): Raw model output to parse.

    Returns:
        Optional[str]: Matched HTML relative path, or ``None`` when not found.
    """
    return _first_match(_HTML_PATTERN, txt)


def parse_generated_code(txt: Any) -> Optional[str]:
    """Extract Python code from a markdown code block.

    Args:
        txt (Any): Raw model output that may contain fenced code blocks.

    Returns:
        Optional[str]: Code content from the first Python block, or first generic
        block as fallback, or ``None`` if no block is present.
    """
    raw_text = str(txt)
    python_match = _first_match(_PYTHON_BLOCK_PATTERN, raw_text)
    if python_match:
        return python_match.strip()

    generic_match = _first_match(_GENERIC_BLOCK_PATTERN, raw_text)
    if generic_match:
        return generic_match.strip()

    return None


def assert_png_file(txt: Any) -> Optional[str]:
    """Validate that output contains a PNG file path in ``./tmp_imgs/``.

    Args:
        txt (Any): Output object to validate.

    Returns:
        Optional[str]: Error message when invalid, otherwise ``None``.
    """
    path = parse_output_img(txt)
    if path is None:
        return f"Function should return a PNG file path in ./tmp_imgs/, but got: {txt}"
    return None


def assert_html_file(txt: Any) -> Optional[str]:
    """Validate that output contains an HTML file path in ``./tmp_imgs/``.

    Args:
        txt (Any): Output object to validate.

    Returns:
        Optional[str]: Error message when invalid, otherwise ``None``.
    """
    path = parse_output_html(txt)
    if path is None:
        return f"Function should return an HTML file path in ./tmp_imgs/, but got: {txt}"
    return None


def assert_pd(obj: Any) -> Optional[str]:
    """Validate that an object is a pandas DataFrame.

    Args:
        obj (Any): Object to validate.

    Returns:
        Optional[str]: Error message when invalid, otherwise ``None``.
    """
    if not isinstance(obj, pandas.DataFrame):
        return f"Expected result type {pandas.DataFrame.__name__}, but got {type(obj)}."
    return None


def assert_str(obj: Any) -> Optional[str]:
    """Validate that an object is a string.

    Args:
        obj (Any): Object to validate.

    Returns:
        Optional[str]: Error message when invalid, otherwise ``None``.
    """
    if not isinstance(obj, str):
        return f"Expected result type {str.__name__}, but got {type(obj)}."
    return None
