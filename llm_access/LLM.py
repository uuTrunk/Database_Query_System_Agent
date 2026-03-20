from typing import Any

import dashscope

from config.get_config import config_data
from llm_access import get_api
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_llm() -> Any:
    """Create and return a configured language model client.

    Args:
        None.

    Returns:
        Any: A model client instance compatible with ``prompt | llm`` invocation.

    Raises:
        ValueError: If ``llm.model_provider`` is unsupported.
    """
    model_provider = str(config_data["llm"]["model_provider"]).strip().lower()
    model_name = config_data["llm"]["model"]

    if model_provider == "qwen":
        from langchain_community.llms import Tongyi

        api_key = get_api.get_api_key_from_file()
        dashscope.api_key = api_key
        logger.info("Initialized Tongyi model: %s", model_name)
        return Tongyi(dashscope_api_key=api_key, model_name=model_name)

    if model_provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Initialized OpenAI-compatible model: %s", model_name)
        return ChatOpenAI(
            temperature=0.95,
            model=model_name,
            openai_api_key=get_api.get_api_key_from_file("./llm_access/api_key_openai.txt"),
            openai_api_base=config_data["llm"].get("url", ""),
        )

    raise ValueError(
        "Unsupported llm.model_provider value: "
        f"{config_data['llm']['model_provider']}. Expected one of: qwen, openai."
    )
