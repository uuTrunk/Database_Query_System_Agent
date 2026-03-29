from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from config.get_config import config_data
from llm_access import call_llm_test
from pgv.ask import build_semantic_context
from utils.logger import setup_logger
from utils.output_parsing import parse_output

logger = setup_logger(__name__)

pd.set_option("display.max_columns", None)


def _append_semantic_context(all_prompt: str, question: str) -> str:
    """Append vector-retrieved semantic hints to prompt text.

    Args:
        all_prompt (str): Prompt text assembled from schema sample and constraints.
        question (str): Original user question used for semantic retrieval.

    Returns:
        str: Prompt text with semantic hints appended when available.
    """
    try:
        semantic_context = build_semantic_context(question)
    except Exception as exc:
        logger.warning("Semantic context retrieval failed: %s", exc)
        return all_prompt

    if not semantic_context:
        return all_prompt

    return all_prompt + "\n" + semantic_context


def _slice_dfs(df_dict: Dict[str, pd.DataFrame], lines: int) -> Dict[str, pd.DataFrame]:
    """Create a lightweight preview of each DataFrame.

    Args:
        df_dict (dict[str, pandas.DataFrame]): Mapping from table name to DataFrame.
        lines (int): Maximum number of rows retained per table preview.

    Returns:
        dict[str, pandas.DataFrame]: Mapping where each DataFrame is trimmed to ``head(lines)``.
    """
    top_dict: Dict[str, pd.DataFrame] = {}
    for key, df in df_dict.items():
        top_dict[key] = df.head(min(lines, len(df)))
    return top_dict


def _build_execution_feedback(
    error: Exception,
    ans_code: str,
    data: List[Any],
) -> Tuple[str, str]:
    """Build retry feedback with executed code and runtime error details.

    Args:
        error (Exception): Runtime exception raised by generated code.
        ans_code (str): Generated Python code that was executed.
        data (list[Any]): Data payload used in execution.

    Returns:
        tuple[str, str]: ``(wrong_code, error_msg)`` fragments appended to retry prompt.
    """
    wrong_code = "\nThe code was executed:```python\n" + ans_code + "\n```"
    error_msg = f"\nThe code raised {type(error).__name__}: {error}"

    if isinstance(error, KeyError):
        available_tables = []
        if data and isinstance(data[0], dict):
            available_tables = list(data[0].keys())
        error_msg += (
            "\nA KeyError indicates your generated code referenced a missing key/column. "
            "You must use exact table keys from dataframes_dict and exact column names from those tables. "
            "Do not invent table names from examples. "
            f"Available table keys: {available_tables}."
        )
        if data and isinstance(data[0], dict):
            columns_str = "; ".join(
                f"{table_name}: {list(table_df.columns)[:10]}{'...' if len(table_df.columns) > 10 else ''}"
                for table_name, table_df in list(data[0].items())[:5]
            )
            if columns_str:
                error_msg += f"\nFirst columns preview: {columns_str}"
        if len(data) > 1:
            error_msg += f"\nForeign key constraints: {data[1]}"

    error_msg += (
        "\nIf you used merge/join, handle duplicate columns with suffixes explicitly "
        "and regenerate the full function."
    )
    return wrong_code, error_msg


def _execute_generated_code(ans_code: str, data_dict: Dict[str, pd.DataFrame]) -> Any:
    """Execute generated Python code and call ``process_data``.

    Args:
        ans_code (str): Generated Python function code.
        data_dict (dict[str, pandas.DataFrame]): DataFrame dictionary passed to ``process_data``.

    Returns:
        Any: Return value from ``process_data``.

    Raises:
        ValueError: If ``process_data`` is missing or not callable.
        Exception: Propagates any exception raised by generated code.
    """
    execution_namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(ans_code, execution_namespace, execution_namespace)
    process_data = execution_namespace.get("process_data")
    if not callable(process_data):
        raise ValueError("Generated code must define a callable function named process_data.")
    return process_data(data_dict)





def get_final_prompt(data: List[Any], question: str) -> str:
    """Assemble the final prompt sent to the model for Python code generation.

    Args:
        data (list[Any]): Input payload where ``data[0]`` is a DataFrame dict,
            ``data[1]`` optionally contains key constraints, and ``data[2]`` optionally contains comments.
        question (str): Task instruction text appended to the prompt template.

    Returns:
        str: Fully composed prompt including sampled data, optional key constraints,
        optional comments, optional semantic vector hints, and output format constraints.
    """
    data_rows = int(config_data["ai"]["data_rows"])
    data_dict: Dict[str, pd.DataFrame] = {}
    if data and isinstance(data[0], dict):
        data_dict = data[0]

    data_slice = _slice_dfs(data_dict, max(1, data_rows))
    pre_prompt = (
        "Write a Python function called process_data that takes only a pandas "
        "dataframe dictionary called dataframes_dict as input and performs the following operations:\n"
    )

    data_prompt = (
        "Here is the dataframe dictionary sample "
        "(it is only structural sample data, not full production data):\n"
    )

    key_prompt = (
        "The tables may be connected through columns with different names, "
        "and those relations are defined as key constraints.\n"
        "Format: {'table_name': {'table_name.column': 'referred_table.referred_column'}}\n"
    )

    comment_prompt = "Here are table and column comments for semantic reference:\n"

    end_prompt = (
        "Return code in a single markdown code block only. "
        "Do not include explanations, shell commands, or comments. "
        "Do not call the function. Do not print anything. "
        "Import required modules inside the function. "
        "Do not mock data. "
        "Apply query optimization principles: filter rows as early as possible before merge/groupby/sort, "
        "select only required columns before joins, avoid Cartesian products, and avoid unnecessary full-table sorting."
    )

    all_prompt = pre_prompt + question + "\n" + data_prompt + str(data_slice)
    if len(data) > 1 and data[1]:
        all_prompt += "\n" + key_prompt + str(data[1])
    if len(data) > 2 and data[2]:
        all_prompt += "\n" + comment_prompt + str(data[2])

    all_prompt = _append_semantic_context(all_prompt, question)
    all_prompt += "\n" + end_prompt
    return all_prompt


def ask(
    data: List[Any],
    question: str,
    llm: Any,
    assert_func: Callable[[Any], Optional[str]],
    retries: int = 0,
):
    """Run model-generated Python code with retry-on-failure feedback.

    Args:
        data (list[Any]): Input payload where ``data[0]`` is passed to ``process_data``.
        question (str): Task instruction used to build the final prompt.
        llm (Any): Language model used to generate code.
        assert_func (Callable[[Any], Optional[str]]): Validation callback that returns
            ``None`` when output is valid, or an error message otherwise.
        retries (int, optional): Maximum number of regeneration retries after failures.

    Returns:
        tuple[Any, int, str]: ``(result, retries_used, all_prompt)`` where ``result`` is
        ``None`` when all attempts fail.
    """
    all_prompt = get_final_prompt(data, question)
    max_retries = max(0, int(retries))
    wrong_code = ""
    error_msg = ""
    attempts_used = 0

    data_dict: Dict[str, pd.DataFrame] = {}
    if data and isinstance(data[0], dict):
        data_dict = data[0]

    for attempt_index in range(max_retries + 1):
        attempts_used = attempt_index
        prompt_with_feedback = all_prompt + wrong_code + error_msg
        answer_text = call_llm_test.call_llm(prompt_with_feedback, llm)
        ans_code = parse_output.parse_generated_code(answer_text)

        if not ans_code:
            wrong_code = ""
            error_msg = (
                "\nCode must be returned inside a markdown code block, for example:\n"
                "```python\n"
                "def process_data(dataframes_dict):\n"
                "    ...\n"
                "```"
            )
            logger.warning("No code block generated on attempt %s", attempt_index + 1)
            continue

        try:
            result = _execute_generated_code(ans_code, data_dict)
            assert_result = assert_func(result)
            if assert_result:
                raise ValueError(assert_result)
            return result, attempts_used, all_prompt
        except Exception as exc:
            wrong_code, error_msg = _build_execution_feedback(exc, ans_code, data)
            logger.warning(
                "Generated code failed on attempt %s/%s with error: %s",
                attempt_index + 1,
                max_retries + 1,
                exc,
            )

    logger.error("Generation failed after %s attempts", max_retries + 1)
    return None, attempts_used, all_prompt
