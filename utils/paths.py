from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"
APP_LOG_FILE = PROJECT_ROOT / "ask_ai.log"
TMP_IMGS_DIR = PROJECT_ROOT / "tmp_imgs"

DEFAULT_SERVER_PORT = 8007


def ensure_directories(paths: Iterable[PathLike]) -> None:
    """Create directories when they do not exist.

    Args:
        paths (Iterable[PathLike]): Directory paths to create.

    Returns:
        None: This function only creates directories as a side effect.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_runtime_directories() -> None:
    """Create runtime directories required by the service.

    Args:
        None.

    Returns:
        None: This function ensures all runtime folders exist.
    """
    ensure_directories([TMP_IMGS_DIR])
