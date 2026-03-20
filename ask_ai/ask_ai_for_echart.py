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


def get_ask_echart_block_prompt(req):
    """Build the prompt for generating inline ECharts HTML content.

    Args:
        req (Any): Request object containing at least `question`.

    Returns:
        str: Full prompt text instructing generation of an HTML string (not a file path).
    """
    question = req.question
    graph_type = """
        use pyecharts 2.0. the Python function should only return a string of html. do not save it.
        please choose different graph type based on the question, do not always use bar. 
        no graph title no set theme! no theme! no theme ! 
        """

    example_code = """
        here is an example: 
        ```python
        def process_data(dataframes_dict):
            import pandas as pd
            import math
            from pyecharts import #
            # do not set theme!!!
            # generate code to perform operations here

            html_string = chart.render_notebook().data # this returns a html string
            return html_string
        ```
        """
    return question + graph_type + example_code


def ask_echart_block(data, req, llm):
    """Generate ECharts HTML strings concurrently and return the first valid result.

    Args:
        data (list): Input payload consumed by downstream prompt assembly and execution.
        req (Any): Request object containing `concurrent` and `retries` settings.
        llm (Any): Language model used for code generation.

    Returns:
        tuple: `(html_str, retries_used, all_prompt, success_ratio)` where `html_str` is `None` on failure.
    """
    max_rounds = _safe_int(config_data["ai"].get("tries"), default=1, minimum=1)
    wait_threshold = _safe_int(config_data["ai"].get("wait"), default=1, minimum=1)
    concurrent_workers = _safe_int(getattr(req, "concurrent", 1), default=1, minimum=1)
    retries = _safe_int(getattr(req, "retries", 0), default=0, minimum=0)

    prompt = get_ask_echart_block_prompt(req)
    last_retries_used = 0
    last_prompt = ""

    for round_index in range(max_rounds):
        result_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [
                executor.submit(
                    ask_api.ask,
                    data,
                    prompt,
                    llm,
                    parse_output.assert_str,
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
                    logger.exception("EChart-block generation worker failed: %s", exc)
                    continue

                if result is not None:
                    result_list.append((result, retries_used, all_prompt))
                    if len(result_list) >= wait_threshold:
                        break

        if result_list:
            best_result, best_retries_used, best_prompt = result_list[0]
            success_ratio = len(result_list) / float(concurrent_workers)
            return best_result, best_retries_used, best_prompt, success_ratio

        logger.warning("EChart block generation round %s/%s failed", round_index + 1, max_rounds)

    return None, last_retries_used, last_prompt, 0.0


def get_ask_echart_file_prompt(req, tmp_file=False):
    """Build the prompt for generating an ECharts HTML file.

    Args:
        req (Any): Request object containing at least `question`.
        tmp_file (bool, optional): Whether to force a fixed temporary output path. Defaults to False.

    Returns:
        str: Full prompt text instructing code to render and return an HTML file path.
    """
    question = req.question
    graph_type = """
            use pyecharts 2.0. the Python function should return a string file path in ./tmp_imgs/ only 
            and the graph html generated should be stored in that path. 
            please choose different graph type based on the question, do not always use bar.  
            no graph title no set theme! no theme! no theme ! 
            file path must be:
            """

    example_code = """
            here is an example: 
            ```python
            def process_data(dataframes_dict):
                import pandas as pd
                import math
                from pyecharts import #
                # generate code to perform operations here
                chart.render(file_path)
                return file_path
            ```
            """
    if not tmp_file:
        return question + graph_type + path_tools.generate_html_path() + example_code
    else:
        return question + graph_type + "./tmp_imgs/tmp.html" + example_code


def ask_echart_file(data, req, llm):
    """Generate ECharts HTML file paths concurrently and return the first valid result.

    Args:
        data (list): Input payload consumed by downstream prompt assembly and execution.
        req (Any): Request object containing `concurrent` and `retries` settings.
        llm (Any): Language model used for code generation.

    Returns:
        tuple: `(html_path, retries_used, all_prompt, success_ratio)` where `html_path` is `None` on failure.
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
                    get_ask_echart_file_prompt(req),
                    llm,
                    parse_output.assert_html_file,
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
                    logger.exception("EChart-file generation worker failed: %s", exc)
                    continue

                graph_path = parse_output.parse_output_html(result)
                if graph_path is not None:
                    result_list.append((graph_path, retries_used, all_prompt))
                    if len(result_list) >= wait_threshold:
                        break

        if result_list:
            best_path, best_retries_used, best_prompt = result_list[0]
            success_ratio = len(result_list) / float(concurrent_workers)
            return best_path, best_retries_used, best_prompt, success_ratio

        logger.warning("EChart file generation round %s/%s failed", round_index + 1, max_rounds)

    return None, last_retries_used, last_prompt, 0.0
