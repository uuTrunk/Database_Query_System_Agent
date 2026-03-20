from typing import Any

from langchain.globals import set_llm_cache
from langchain_core.prompts import PromptTemplate

set_llm_cache(None)


def _normalize_llm_output(raw_output: Any) -> str:
    """Normalize provider-specific model output into a plain text string.

    Args:
        raw_output (Any): Raw response object returned by LangChain model invocation.

    Returns:
        str: Text content extracted from the response.
    """
    if hasattr(raw_output, "content"):
        return str(getattr(raw_output, "content"))
    return str(raw_output)


def call_llm(question: str, llm: Any) -> str:
    """Invoke a language model with a plain single-variable prompt.

    Args:
        question (str): Prompt text sent to the language model.
        llm (Any): Model instance compatible with ``PromptTemplate | llm`` invocation.

    Returns:
        str: Plain text model output.
    """
    prompt = PromptTemplate(template="{question}", input_variables=["question"])
    llm_chain = prompt | llm
    raw_output = llm_chain.invoke(question)
    return _normalize_llm_output(raw_output)
