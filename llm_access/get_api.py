from pathlib import Path


def get_api_key_from_file(file_path: str = "./llm_access/api_key_qwen.txt") -> str:
    """Read an API key from a local text file.

    Args:
        file_path (str, optional): Relative or absolute path to the API key file.

    Returns:
        str: API key text with surrounding whitespace removed.

    Raises:
        FileNotFoundError: If the key file does not exist.
        ValueError: If the key file is empty.
    """
    api_key_path = Path(file_path)
    if not api_key_path.exists():
        raise FileNotFoundError(f"API key file not found: {api_key_path}")

    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {api_key_path}")

    return api_key

