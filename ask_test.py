import os
import shutil
from typing import Any, Dict, Optional, Tuple

import http.client
import json

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8007


def load_server_target() -> Tuple[str, int]:
    """Load API host and port from project config with safe fallbacks.

    Args:
        None.

    Returns:
        tuple[str, int]: Server host and port used by this script.
    """
    try:
        from config.get_config import config_data

        host = str(config_data.get("server", {}).get("host", DEFAULT_SERVER_HOST))
        port = int(config_data.get("server_port", DEFAULT_SERVER_PORT))
        return host, port
    except Exception:
        return DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT


SERVER_HOST, SERVER_PORT = load_server_target()


def send_request(request_data: Dict[str, Any]) -> Tuple[Optional[int], bytes | str]:
    """Send a POST request to the local `/ask/echart-file-2` endpoint.

    Args:
        request_data (dict[str, Any]): JSON-serializable request payload.

    Returns:
        tuple[Optional[int], bytes | str]: `(status_code, response_body)` on success,
        or `(None, error_message)` on network/connection failure.
    """
    json_data = json.dumps(request_data)
    headers = {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection(f"{SERVER_HOST}:{SERVER_PORT}")
    try:
        conn.request("POST", "/ask/echart-file-2", body=json_data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


def parse_response_json(raw_response: bytes | str) -> Dict[str, Any]:
    """Parse API response into JSON with clear fallback errors.

    Args:
        raw_response (bytes | str): Raw response returned by `send_request`.

    Returns:
        dict[str, Any]: Parsed JSON object.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    if isinstance(raw_response, bytes):
        text = raw_response.decode("utf-8", errors="replace")
    else:
        text = raw_response

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {text}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, but got: {type(parsed)}")

    return parsed


if __name__ == "__main__":
    q = "列出销量大于4000的所有国家的名称"
    request = {
        "question": q,
        "concurrent": [1, 1],
        "retries": [2, 2]
    }
    print(f"Request target: http://{SERVER_HOST}:{SERVER_PORT}/ask/echart-file-2")
    status_code, response = send_request(request)
    print(status_code)
    if status_code is None:
        print(f"Request failed before receiving HTTP response: {response}")
        raise SystemExit(1)

    try:
        response_data = parse_response_json(response)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)

    print(response_data)

    output_file = response_data.get("file", "")
    if not output_file:
        print("No output file path returned by server.")
        raise SystemExit(1)

    print(output_file)

    # 检查文件是否存在
    if os.path.exists(output_file):
        # 获取文件名和扩展名
        file_name = os.path.basename(output_file)
        file_extension = os.path.splitext(file_name)[1]

        # 构建新的文件名
        new_file_name = f"temp{file_extension}"

        # 复制文件到当前目录并重命名
        shutil.copy(output_file, os.path.join(os.getcwd(), new_file_name))
        print(f"File copied and renamed to {new_file_name}")
    else:
        print(f"Output file does not exist: {output_file}")
        raise SystemExit(1)
