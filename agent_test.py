import base64
import http.client
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8007


def load_server_target() -> Tuple[str, int]:
    """Load API host/port from config with safe fallbacks.

    Returns:
        tuple[str, int]: (host, port)
    """
    try:
        from config.get_config import config_data

        server = config_data.get("server", {}) if isinstance(config_data, dict) else {}
        host = str(server.get("host", DEFAULT_SERVER_HOST))
        port = int(config_data.get("server_port", DEFAULT_SERVER_PORT))
        return host, port
    except Exception:
        return DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT


def send_graph_steps_request(host: str, port: int, request_data: Dict[str, Any]) -> Tuple[Optional[int], bytes | str]:
    """Send POST /ask/graph-steps request.

    Args:
        host: API host.
        port: API port.
        request_data: JSON payload.

    Returns:
        tuple[Optional[int], bytes | str]: (status_code, response_body) on success,
        or (None, error_message) on request failure.
    """
    conn = http.client.HTTPConnection(host, port)
    headers = {"Content-Type": "application/json"}
    body = json.dumps(request_data)

    try:
        conn.request("POST", "/ask/graph-steps", body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


def parse_response_json(raw_response: bytes | str) -> Dict[str, Any]:
    """Parse response payload into a JSON object.

    Args:
        raw_response: Raw bytes or text response.

    Returns:
        dict[str, Any]: Parsed response JSON.

    Raises:
        ValueError: If response is not a valid JSON object.
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
        raise ValueError(f"Expected JSON object, got: {type(parsed)}")

    return parsed


def save_image_if_present(response_json: Dict[str, Any], output_path: Path) -> bool:
    """Save base64 image payload to local file when present.

    Args:
        response_json: Parsed API response.
        output_path: Local file path for image output.

    Returns:
        bool: True if image was saved, False otherwise.
    """
    image_b64 = response_json.get("image_data")
    if not image_b64 or not isinstance(image_b64, str):
        return False

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return True


def main() -> None:
    host, port = load_server_target()

    request_data = {
        "question": "Show the top 10 most populous cities in China as a bar chart.",
        "concurrent": [1, 1],
        "retries": [2, 2],
    }

    print(f"Request target: http://{host}:{port}/ask/graph-steps")
    status_code, response = send_graph_steps_request(host, port, request_data)

    if status_code is None:
        print(f"Request failed before HTTP response: {response}")
        raise SystemExit(1)

    print(f"HTTP status: {status_code}")

    try:
        response_json = parse_response_json(response)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)

    api_code = response_json.get("code")
    print(f"API code: {api_code}")

    if status_code != 200 or api_code != 200:
        print("Request did not succeed.")
        print(json.dumps(response_json, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    image_saved = save_image_if_present(response_json, Path("tmp_imgs") / "agent_test_graph_steps.png")

    print("/ask/graph-steps test passed.")
    print(f"retries_used: {response_json.get('retries_used')}")
    print(f"remote file path: {response_json.get('file')}")
    print(f"local image saved: {image_saved}")


if __name__ == "__main__":
    main()
