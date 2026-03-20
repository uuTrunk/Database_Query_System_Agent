import concurrent
from typing import Any, Dict, Tuple

from ask_ai import ask_api
from config.get_config import config_data
from utils.logger import setup_logger
from utils.output_parsing import parse_output

logger = setup_logger(__name__)


def _safe_int(value: Any, default: int, minimum: int = 0) -> int:
    """Convert an arbitrary value to a bounded integer.

    Args:
        value (Any): Value to convert.
        default (int): Fallback value when conversion fails.
        minimum (int, optional): Lower bound applied to the converted value.

    Returns:
        int: Converted integer value, clamped to ``minimum``.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def get_ask_pd_prompt(req):
    """Build the pandas-processing prompt for code generation.

    Args:
        req (Any): Request object containing at least `question`.

    Returns:
        str: Full prompt text describing pandas transformation constraints and examples.
    """
    question = req.question
    example_code = """ 
       the Python function should return a single pandas dataframe only!!! 
       do not draw any graph at this step.
       
       ### CRITICAL RULES FOR MULTI-TABLE JOINS ###
       1. If you join/merge tables that share the same column names (e.g., 'Population' in both 'city' and 'country'), 
          ALWAYS specify 'suffixes' in the merge function, for example: 
          merged_df = pd.merge(df1, df2, on='ID', suffixes=('_left', '_right'))
       2. NEVER assume a column name exists after a join without considering these suffixes.
       3. Use the renamed columns (like 'Population_left') in subsequent filters or selections.
       
       here is an example: 
       ```python
       def process_data(dataframes_dict):
           import pandas as pd
           city = dataframes_dict['city']
           country = dataframes_dict['country']
           # Explicitly handle suffixes to avoid KeyError
           merged = pd.merge(city, country, left_on='CountryCode', right_on='Code', suffixes=('_city', '_country'))
           # Use the explicit column name
           result = merged.nsmallest(5, 'Population_city')[['Name_city', 'Population_city']]
           return result
       ```
       """
    return question + example_code


def ask_pd(data, req, llm):
    """Generate pandas transformation results concurrently and return a validated result.

    Args:
        data (list): Input payload consumed by downstream prompt assembly and execution.
        req (Any): Request object containing `concurrent` and `retries` settings.
        llm (Any): Language model used for code generation.

    Returns:
        tuple: `(result_df, retries_used, all_prompt, success_ratio)` where `result_df` is `None` on failure.
    """
    max_rounds = _safe_int(config_data["ai"].get("tries"), default=1, minimum=1)
    wait_threshold = _safe_int(config_data["ai"].get("wait"), default=1, minimum=1)
    concurrent_workers = _safe_int(getattr(req, "concurrent", 1), default=1, minimum=1)
    retries = _safe_int(getattr(req, "retries", 0), default=0, minimum=0)

    prompt = get_ask_pd_prompt(req)
    last_retries_used = 0
    last_prompt = ""

    for round_index in range(max_rounds):
        clean_data_pd_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(
                    ask_api.ask,
                    data,
                    prompt,
                    llm,
                    parse_output.assert_pd,
                    retries,
                )
                for _ in range(concurrent_workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result, retries_used, all_prompt = future.result()
                    last_retries_used = retries_used
                    last_prompt = all_prompt
                except Exception as exc:
                    logger.exception("PD generation worker failed: %s", exc)
                    continue

                if result is not None:
                    clean_data_pd_list.append((result, retries_used, all_prompt))
                    if len(clean_data_pd_list) >= wait_threshold:
                        break

        if clean_data_pd_list:
            best_result, best_retries_used, best_prompt = clean_data_pd_list[0]
            success_ratio = len(clean_data_pd_list) / float(concurrent_workers)
            return best_result, best_retries_used, best_prompt, success_ratio

        logger.warning("PD generation round %s/%s failed", round_index + 1, max_rounds)

    return None, last_retries_used, last_prompt, 0.0
