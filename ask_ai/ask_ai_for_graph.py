import concurrent
from typing import Any

from ask_ai import ask_api
from ask_ai import input_process
from config.get_config import config_data
from utils import path_tools
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


def get_ask_graph_prompt(req, llm, tmp_file=False, img_type=True):
    """Build the graph-generation prompt sent to the code-generation agent.

    Args:
        req (Any): Request object containing at least `question`.
        llm (Any): Language model used to infer chart type when `img_type` is enabled.
        tmp_file (bool, optional): Whether to force a fixed temporary output path. Defaults to False.
        img_type (bool, optional): Whether to prepend an LLM-selected chart type hint. Defaults to True.

    Returns:
        str: Full prompt text instructing code generation for matplotlib image output.
    """
    question = req.question
    graph_type = """
        use matplotlib. the Python function should return a string file path in ./tmp_imgs/ only 
        and the image generated should be stored in that path. 
        file path must be:
        """
    if img_type:
        graph_type = input_process.get_chart_type(question, llm) + graph_type
    example_code = """
        here is an example: 
        ```python
        def process_data(dataframes_dict):
            import pandas as pd
            import math
            import matplotlib.pyplot as plt
            import matplotlib
            import PIL
            # generate code to perform operations here
            return path
        ```
        """
    if not tmp_file:
        return question + graph_type + path_tools.generate_img_path() + example_code
    else:
        return question + graph_type + "./tmp_imgs/tmp.png" + example_code


def ask_graph(data, req, llm):
    """Generate graph file paths concurrently and return the first accepted result.

    Args:
        data (list): Input payload consumed by downstream prompt assembly and execution.
        req (Any): Request object containing `concurrent` and `retries` settings.
        llm (Any): Language model used for code generation.

    Returns:
        tuple: `(img_path, retries_used, all_prompt, success_ratio)` where `img_path` is `None` on failure.
    """
    max_rounds = _safe_int(config_data["ai"].get("tries"), default=1, minimum=1)
    wait_threshold = _safe_int(config_data["ai"].get("wait"), default=1, minimum=1)
    concurrent_workers = _safe_int(getattr(req, "concurrent", 1), default=1, minimum=1)
    retries = _safe_int(getattr(req, "retries", 0), default=0, minimum=0)

    last_retries_used = 0
    last_prompt = ""

    for round_index in range(max_rounds):
        result_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(
                    ask_api.ask,
                    data,
                    get_ask_graph_prompt(req, llm),
                    llm,
                    parse_output.assert_png_file,
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
                    logger.exception("Graph generation worker failed: %s", exc)
                    continue

                img_path = parse_output.parse_output_img(result)
                if img_path is not None:
                    result_list.append((img_path, retries_used, all_prompt))
                    if len(result_list) >= wait_threshold:
                        break

        if result_list:
            best_path, best_retries_used, best_prompt = result_list[0]
            success_ratio = len(result_list) / float(concurrent_workers)
            return best_path, best_retries_used, best_prompt, success_ratio

        logger.warning("Graph generation round %s/%s failed", round_index + 1, max_rounds)

    return None, last_retries_used, last_prompt, 0.0
