import secrets
import string
from pathlib import Path

from utils.paths import TMP_IMGS_DIR, ensure_runtime_directories


def generate_random_string(length: int = 8) -> str:
    """Generate a random lowercase string.

    Args:
        length (int, optional): Number of characters to generate.

    Returns:
        str: Random string composed of lowercase ASCII letters.
    """
    letters = string.ascii_lowercase
    return "".join(secrets.choice(letters) for _ in range(length))


def _build_tmp_path(extension: str) -> str:
    """Build a random temporary file path under the runtime output directory.

    Args:
        extension (str): File extension including dot, for example ``.png``.

    Returns:
        str: Relative path string compatible with existing downstream logic.
    """
    ensure_runtime_directories()
    file_path = TMP_IMGS_DIR / f"{generate_random_string()}{extension}"

    project_root = TMP_IMGS_DIR.parent
    relative_path = Path(file_path).relative_to(project_root)
    return f"./{relative_path.as_posix()}"


def generate_img_path() -> str:
    """Generate a random PNG file path under ``./tmp_imgs/``.

    Args:
        None.

    Returns:
        str: Relative PNG path with a random filename.
    """
    return _build_tmp_path(".png")


def generate_html_path() -> str:
    """Generate a random HTML file path under ``./tmp_imgs/``.

    Args:
        None.

    Returns:
        str: Relative HTML path with a random filename.
    """
    return _build_tmp_path(".html")
