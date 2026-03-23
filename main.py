import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# This must be set before importing matplotlib.pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ask_ai import ask_ai_for_echart, ask_ai_for_graph, ask_ai_for_pd, ask_api
import data_access.read_db
from config.get_config import config_data
from llm_access.LLM import get_llm
from pgv.ask import sync_schema_knowledge
from utils import path_tools
from utils.logger import setup_logger
from utils.manuel_mode import pandas_html
from utils.paths import APP_LOG_FILE, ensure_runtime_directories

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

logger = setup_logger(__name__, log_file=str(APP_LOG_FILE), level=logging.INFO)
ensure_runtime_directories()

llm = get_llm()

FOLLOW_UP_VISUAL_QUESTION = (
    "Data in the given dataframe is already filtered. "
    "Please draw a suitable graph based on the provided data."
)

app = FastAPI(title="data-copilot-v2", version="2.0")


class AskRequest(BaseModel):
    """Request model for single-stage generation endpoints.

    Attributes:
        question (str): Natural language question sent by the user.
        concurrent (int): Number of independent concurrent generations.
        retries (int): Retry count for each single generation chain.
    """

    question: str = Field(..., description="Natural language question.")
    concurrent: int = Field(1, ge=1, description="Concurrent request count.")
    retries: int = Field(0, ge=0, description="Retry times for one request.")


class AskRequestSteps(BaseModel):
    """Request model for two-stage generation endpoints.

    Attributes:
        question (str): User question used in stage one (data filtering).
        concurrent (list[int]): Two-item concurrency list for stages 1 and 2.
        retries (list[int]): Two-item retry list for stages 1 and 2.
    """

    question: str = Field(..., description="Natural language question.")
    concurrent: List[int] = Field(..., min_length=2, max_length=2)
    retries: List[int] = Field(..., min_length=2, max_length=2)

    @field_validator("concurrent")
    @classmethod
    def validate_concurrent(cls, value: List[int]) -> List[int]:
        """Validate stage concurrency list.

        Args:
            value (list[int]): Candidate concurrency values for each stage.

        Returns:
            list[int]: Validated positive integer list with exactly two items.

        Raises:
            ValueError: If any value is smaller than 1.
        """
        normalized = [int(item) for item in value]
        if any(item < 1 for item in normalized):
            raise ValueError("All concurrent values must be >= 1.")
        return normalized

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, value: List[int]) -> List[int]:
        """Validate stage retry list.

        Args:
            value (list[int]): Candidate retry values for each stage.

        Returns:
            list[int]: Validated non-negative integer list with exactly two items.

        Raises:
            ValueError: If any value is negative.
        """
        normalized = [int(item) for item in value]
        if any(item < 0 for item in normalized):
            raise ValueError("All retries values must be >= 0.")
        return normalized


def fetch_data(force_reload: bool = False) -> List[Any]:
    """Fetch table data and metadata used by prompt construction.

    Args:
        force_reload (bool, optional): Whether to force a database reload and bypass cache.

    Returns:
        list[Any]: Three-item payload ``[tables_data, foreign_keys, comments]``.

    Raises:
        RuntimeError: If database data cannot be loaded.
    """
    try:
        data, key, comment = data_access.read_db.get_data_from_db(force_reload=force_reload)
        payload = [data, key, comment]

        try:
            sync_schema_knowledge(payload, force_rebuild=force_reload)
        except Exception as sync_exc:
            logger.warning("Vector schema sync skipped due to error: %s", sync_exc)

        return payload
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch database data: {exc}") from exc


def _build_failure_response(
    retries_used: Any,
    prompt: Any,
    msg: str = "gen failed",
    success: Any = 0.0,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardized failure response body.

    Args:
        retries_used (Any): Retry usage data for request or stages.
        prompt (Any): Prompt text or list of prompts used in generation.
        msg (str, optional): Failure message.
        success (Any, optional): Success score value or list.
        extra_fields (Optional[dict[str, Any]], optional): Additional fields to include.

    Returns:
        dict[str, Any]: Unified failure response dictionary.
    """
    payload = {
        "code": 504,
        "retries_used": retries_used,
        "msg": msg,
        "prompt": prompt,
        "success": success,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _write_text_file(file_path: str, content: str) -> str:
    """Write UTF-8 text content to a file.

    Args:
        file_path (str): Output file path.
        content (str): Text content to write.

    Returns:
        str: The same ``file_path`` value for chaining.
    """
    Path(file_path).write_text(content, encoding="utf-8")
    return file_path


def _read_text_file(file_path: str) -> str:
    """Read UTF-8 text content from a file.

    Args:
        file_path (str): Input file path.

    Returns:
        str: File content as UTF-8 string.
    """
    return Path(file_path).read_text(encoding="utf-8")


def _read_binary_as_base64(file_path: str) -> str:
    """Read binary file content and encode it as base64 text.

    Args:
        file_path (str): Binary file path.

    Returns:
        str: Base64-encoded text representation.
    """
    binary_data = Path(file_path).read_bytes()
    return base64.b64encode(binary_data).decode("utf-8")


def _build_step_requests(original_request: AskRequestSteps) -> Tuple[AskRequest, AskRequest]:
    """Build stage request objects for two-step pipelines.

    Args:
        original_request (AskRequestSteps): Raw two-stage request body.

    Returns:
        tuple[AskRequest, AskRequest]: Stage-1 and stage-2 request models.
    """
    request_stage_1 = AskRequest(
        question=original_request.question,
        concurrent=original_request.concurrent[0],
        retries=original_request.retries[0],
    )
    request_stage_2 = AskRequest(
        question=FOLLOW_UP_VISUAL_QUESTION,
        concurrent=original_request.concurrent[1],
        retries=original_request.retries[1],
    )
    return request_stage_1, request_stage_2


@app.on_event("startup")
async def on_startup() -> None:
    """Initialize runtime directories and warm up database cache on startup.

    Args:
        None.

    Returns:
        None: This function performs startup side effects.
    """
    ensure_runtime_directories()
    fetch_data(force_reload=False)
    logger.info("Service startup completed")


@app.post("/ask/pd")
async def ask_pd(request: AskRequest) -> Dict[str, Any]:
    """Generate a pandas-style query result from a natural language question.

    Args:
        request (AskRequest): Request payload containing question, concurrency, and retries.

    Returns:
        dict[str, Any]: Success response with DataFrame dictionary output or standardized failure payload.
    """
    try:
        dict_data = fetch_data()
        result, retries_used, all_prompt, success = ask_ai_for_pd.ask_pd(dict_data, request, llm)
        if result is None:
            return _build_failure_response(
                retries_used=retries_used,
                prompt=all_prompt,
                success=0.0,
                extra_fields={"answer": ""},
            )
        return {
            "code": 200,
            "retries_used": retries_used,
            "answer": result.to_dict(),
            "prompt": all_prompt,
            "success": success,
        }
    except Exception as exc:
        logger.exception("/ask/pd failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/pd-walker")
async def ask_pd_walker(request: AskRequest) -> Dict[str, Any]:
    """Generate a pandas result and render it as HTML content.

    Args:
        request (AskRequest): Request payload containing question, concurrency, and retries.

    Returns:
        dict[str, Any]: Success response with HTML content and file path or standardized failure payload.
    """
    try:
        dict_data = fetch_data()
        result, retries_used, all_prompt, success = ask_ai_for_pd.ask_pd(dict_data, request, llm)
        if result is None:
            return _build_failure_response(
                retries_used=retries_used,
                prompt=all_prompt,
                success=0.0,
                extra_fields={"html": "", "file": ""},
            )

        html_content = pandas_html.get_html(result)
        file_path = path_tools.generate_html_path()
        _write_text_file(file_path, html_content)

        return {
            "code": 200,
            "retries_used": retries_used,
            "html": _read_text_file(file_path),
            "file": file_path,
            "prompt": all_prompt,
            "success": success,
        }
    except Exception as exc:
        logger.exception("/ask/pd-walker failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/graph")
async def ask_graph(request: AskRequest) -> Dict[str, Any]:
    """Generate a static chart image from a natural language question.

    Args:
        request (AskRequest): Request payload containing question, concurrency, and retries.

    Returns:
        dict[str, Any]: Success response with base64 image data or standardized failure payload.
    """
    try:
        dict_data = fetch_data()
        result, retries_used, all_prompt, success = ask_ai_for_graph.ask_graph(dict_data, request, llm)
        if result is None:
            return _build_failure_response(
                retries_used=retries_used,
                prompt=all_prompt,
                success=0.0,
                extra_fields={"image_data": "", "file": ""},
            )

        return {
            "code": 200,
            "retries_used": retries_used,
            "image_data": _read_binary_as_base64(result),
            "file": result,
            "prompt": all_prompt,
            "success": success,
        }
    except Exception as exc:
        logger.exception("/ask/graph failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/graph-steps")
async def ask_graph_steps(original_request: AskRequestSteps) -> Dict[str, Any]:
    """Run a two-stage pipeline: dataframe filtering then static graph generation.

    Args:
        original_request (AskRequestSteps): Two-stage request payload.

    Returns:
        dict[str, Any]: Success response with final image output or standardized failure payload.
    """
    retries_used_1 = 0
    retries_used_2 = 0
    prompt_1 = ""
    prompt_2 = ""

    try:
        dict_data = fetch_data()
        request_stage_1, request_stage_2 = _build_step_requests(original_request)

        result_1, retries_used_1, prompt_1, success_1 = ask_ai_for_pd.ask_pd(dict_data, request_stage_1, llm)
        if result_1 is None:
            return _build_failure_response(
                retries_used=[retries_used_1, retries_used_2],
                prompt=[prompt_1, prompt_2],
                success=[0.0, 0.0],
                extra_fields={"image_data": "", "file": ""},
            )

        result_2, retries_used_2, prompt_2, success_2 = ask_ai_for_graph.ask_graph(
            [{"data": result_1}],
            request_stage_2,
            llm,
        )
        if result_2 is None:
            return _build_failure_response(
                retries_used=[retries_used_1, retries_used_2],
                prompt=[prompt_1, prompt_2],
                success=[0.0, 0.0],
                extra_fields={"image_data": "", "file": ""},
            )

        return {
            "code": 200,
            "retries_used": [retries_used_1, retries_used_2],
            "image_data": _read_binary_as_base64(result_2),
            "file": result_2,
            "prompt": [prompt_1, prompt_2],
            "success": [success_1, success_2],
        }
    except Exception as exc:
        logger.exception("/ask/graph-steps failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/echart-block")
async def ask_echart_block(request: AskRequest) -> Dict[str, Any]:
    """Generate ECharts HTML content directly and persist it to a file.

    Args:
        request (AskRequest): Request payload containing question, concurrency, and retries.

    Returns:
        dict[str, Any]: Success response with HTML block and file path or standardized failure payload.
    """
    try:
        dict_data = fetch_data()
        result, retries_used, all_prompt, success = ask_ai_for_echart.ask_echart_block(dict_data, request, llm)
        if result is None:
            return _build_failure_response(
                retries_used=retries_used,
                prompt=all_prompt,
                success=0.0,
                extra_fields={"html": "", "file": ""},
            )

        file_path = path_tools.generate_html_path()
        _write_text_file(file_path, result)
        return {
            "code": 200,
            "retries_used": retries_used,
            "html": _read_text_file(file_path),
            "file": file_path,
            "prompt": all_prompt,
            "success": success,
        }
    except Exception as exc:
        logger.exception("/ask/echart-block failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/echart-file")
async def ask_echart_file(request: AskRequest) -> Dict[str, Any]:
    """Generate an ECharts HTML file and return its content.

    Args:
        request (AskRequest): Request payload containing question, concurrency, and retries.

    Returns:
        dict[str, Any]: Success response with generated HTML content or standardized failure payload.
    """
    try:
        dict_data = fetch_data()
        result, retries_used, all_prompt, success = ask_ai_for_echart.ask_echart_file(dict_data, request, llm)
        if result is None:
            return _build_failure_response(
                retries_used=retries_used,
                prompt=all_prompt,
                success=0.0,
                extra_fields={"html": "", "file": ""},
            )

        return {
            "code": 200,
            "retries_used": retries_used,
            "html": _read_text_file(result),
            "file": result,
            "prompt": all_prompt,
            "success": success,
        }
    except Exception as exc:
        logger.exception("/ask/echart-file failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/echart-file-2")
async def ask_echart_file_2(original_request: AskRequestSteps) -> Dict[str, Any]:
    """Run a two-stage pipeline: dataframe filtering then ECharts file generation.

    Args:
        original_request (AskRequestSteps): Two-stage request payload.

    Returns:
        dict[str, Any]: Success response with final HTML output or standardized failure payload.
    """
    retries_used_1 = 0
    retries_used_2 = 0
    prompt_1 = ""
    prompt_2 = ""

    try:
        dict_data = fetch_data()
        request_stage_1, request_stage_2 = _build_step_requests(original_request)

        result_1, retries_used_1, prompt_1, success_1 = ask_ai_for_pd.ask_pd(dict_data, request_stage_1, llm)
        if result_1 is None:
            return _build_failure_response(
                retries_used=[retries_used_1, retries_used_2],
                prompt=[prompt_1, prompt_2],
                success=[0.0, 0.0],
                extra_fields={"html": "", "file": ""},
            )

        result_2, retries_used_2, prompt_2, success_2 = ask_ai_for_echart.ask_echart_file(
            [{"data": result_1}],
            request_stage_2,
            llm,
        )
        if result_2 is None:
            return _build_failure_response(
                retries_used=[retries_used_1, retries_used_2],
                prompt=[prompt_1, prompt_2],
                success=[0.0, 0.0],
                extra_fields={"html": "", "file": ""},
            )

        return {
            "code": 200,
            "retries_used": [retries_used_1, retries_used_2],
            "html": _read_text_file(result_2),
            "file": result_2,
            "prompt": [prompt_1, prompt_2],
            "success": [success_1, success_2],
        }
    except Exception as exc:
        logger.exception("/ask/echart-file-2 failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/prompt/pd")
async def prompt_pd(request: AskRequest) -> Dict[str, Any]:
    """Build and return the final prompt used for pandas generation.

    Args:
        request (AskRequest): Request payload containing question and generation settings.

    Returns:
        dict[str, Any]: Prompt preview payload.
    """
    dict_data = fetch_data()
    all_prompt = ask_api.get_final_prompt(dict_data, ask_ai_for_pd.get_ask_pd_prompt(request))
    return {"code": 200, "all_prompt": all_prompt}


@app.post("/prompt/graph")
async def prompt_graph(request: AskRequest) -> Dict[str, Any]:
    """Build and return the final prompt used for static graph generation.

    Args:
        request (AskRequest): Request payload containing question and generation settings.

    Returns:
        dict[str, Any]: Prompt preview payload.
    """
    dict_data = fetch_data()
    all_prompt = ask_api.get_final_prompt(
        dict_data,
        ask_ai_for_graph.get_ask_graph_prompt(
            request,
            llm,
            tmp_file=True,
            img_type=False,
        ),
    )
    return {"code": 200, "all_prompt": all_prompt}


@app.post("/prompt/echart-block")
async def prompt_echart_block(request: AskRequest) -> Dict[str, Any]:
    """Build and return the final prompt used for ECharts block generation.

    Args:
        request (AskRequest): Request payload containing question and generation settings.

    Returns:
        dict[str, Any]: Prompt preview payload.
    """
    dict_data = fetch_data()
    all_prompt = ask_api.get_final_prompt(dict_data, ask_ai_for_echart.get_ask_echart_block_prompt(request))
    return {"code": 200, "all_prompt": all_prompt}


@app.post("/prompt/echart-file")
async def prompt_echart_file(request: AskRequest) -> Dict[str, Any]:
    """Build and return the final prompt used for ECharts file generation.

    Args:
        request (AskRequest): Request payload containing question and generation settings.

    Returns:
        dict[str, Any]: Prompt preview payload.
    """
    dict_data = fetch_data()
    all_prompt = ask_api.get_final_prompt(
        dict_data,
        ask_ai_for_echart.get_ask_echart_file_prompt(request, tmp_file=True),
    )
    return {"code": 200, "all_prompt": all_prompt}


if __name__ == "__main__":
    logger.info("Server starting")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(config_data["server_port"]))
