import base64
import logging
from pathlib import Path
import hashlib
import hmac

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from config.get_config import config_data
from llm_access.LLM import get_llm
from pgv.ask import sync_schema_knowledge
import data_access.read_db
from ask_ai import ask_ai_for_graph, ask_ai_for_pd, ask_api
from utils import path_tools
from utils.logger import setup_logger
from utils.manuel_mode import pandas_html
from utils.paths import APP_LOG_FILE, ensure_runtime_directories
from pydantic import BaseModel, Field, ValidationError

logger = setup_logger(__name__, log_file=str(APP_LOG_FILE), level=logging.INFO)
ensure_runtime_directories()
llm = get_llm()

class AskRequestModel(BaseModel):
    question: str
    concurrent: int = 1
    retries: int = 0

def fetch_data(force_reload=False):
    data, key, comment = data_access.read_db.get_data_from_db(force_reload=force_reload)
    payload = [data, key, comment]
    try:
        sync_schema_knowledge(payload, force_rebuild=force_reload)
    except Exception as sync_exc:
        logger.warning(f"Vector schema sync skipped: {sync_exc}")
    return payload

def _build_failure(retries_used, prompt, msg="gen failed", success=0.0, extra=None):
    r = {"code": 504, "retries_used": retries_used, "msg": msg, "prompt": prompt, "success": success}
    if extra: r.update(extra)
    return r

class LoginView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        auth_config = config_data.get("auth", {})
        LOGIN_USERNAME = str(auth_config.get("username", "admin"))
        LOGIN_PASSWORD = str(auth_config.get("password", ""))
        LOGIN_PASSWORD_SHA256 = str(auth_config.get("password_sha256", "")).strip().lower()

        def _password_is_valid(raw):
            if LOGIN_PASSWORD_SHA256:
                candidate_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest().lower()
                return hmac.compare_digest(candidate_hash, LOGIN_PASSWORD_SHA256)
            return hmac.compare_digest(raw, LOGIN_PASSWORD)

        if hmac.compare_digest(str(username), LOGIN_USERNAME) and _password_is_valid(str(password)):
            return Response({"code": 200, "message": "success"})
        return Response({"code": 401, "message": "invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
        
class AskPdView(APIView):
    def post(self, request):
        try:
            req_data = AskRequestModel(**request.data)
            dict_data = fetch_data()
            result, retries_used, all_prompt, success = ask_ai_for_pd.ask_pd(dict_data, req_data, llm)
            if result is None:
                return Response(
                    _build_failure(retries_used, all_prompt, extra={"answer": ""}),
                    status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
            return Response({
                "code": 200, "retries_used": retries_used, "answer": result.to_dict(),
                "prompt": all_prompt, "success": success
            })
        except Exception as e:
            return Response({"code": 500, "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AskPdWalkerView(APIView):
    def post(self, request):
        try:
            req_data = AskRequestModel(**request.data)
            dict_data = fetch_data()
            result, retries_used, all_prompt, success = ask_ai_for_pd.ask_pd(dict_data, req_data, llm)
            if result is None:
                return Response(
                    _build_failure(retries_used, all_prompt, extra={"html": "", "file": ""}),
                    status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
            html_content = pandas_html.get_html(result)
            file_path = path_tools.generate_html_path()
            Path(file_path).write_text(html_content, encoding="utf-8")
            return Response({
                "code": 200, "retries_used": retries_used, "html": Path(file_path).read_text(encoding="utf-8"),
                "file": file_path, "prompt": all_prompt, "success": success
            })
        except Exception as e:
            return Response({"code": 500, "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AskGraphView(APIView):
    def post(self, request):
        try:
            req_data = AskRequestModel(**request.data)
            dict_data = fetch_data()
            result, retries_used, all_prompt, success = ask_ai_for_graph.ask_graph(dict_data, req_data, llm)
            if result is None:
                return Response(
                    _build_failure(retries_used, all_prompt, extra={"image_data": "", "file": ""}),
                    status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
            b64 = base64.b64encode(Path(result).read_bytes()).decode("utf-8")
            return Response({
                "code": 200, "retries_used": retries_used, "image_data": b64,
                "file": str(result), "prompt": all_prompt, "success": success
            })
        except Exception as e:
            return Response({"code": 500, "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AskGraphStepsView(APIView):
    def post(self, request):
        try:
            data = request.data
            # simplified handling for two stage
            c1, c2 = data.get("concurrent", [1,1])[0:2]
            r1, r2 = data.get("retries", [5,5])[0:2]
            req1 = AskRequestModel(question=data["question"], concurrent=c1, retries=r1)
            req2 = AskRequestModel(question="Data in the given dataframe is already filtered. Please draw a suitable graph based on the provided data.", concurrent=c2, retries=r2)
            
            dict_data = fetch_data()
            res1, ret1, p1, s1 = ask_ai_for_pd.ask_pd(dict_data, req1, llm)
            if res1 is None:
                return Response(
                    _build_failure([ret1, 0], [p1, ""], extra={"image_data": "", "file": ""}),
                    status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
            
            res2, ret2, p2, s2 = ask_ai_for_graph.ask_graph([{"data": res1}], req2, llm)
            if res2 is None:
                return Response(
                    _build_failure([ret1, ret2], [p1, p2], extra={"image_data": "", "file": ""}),
                    status=status.HTTP_504_GATEWAY_TIMEOUT,
                )
                
            b64 = base64.b64encode(Path(res2).read_bytes()).decode("utf-8")
            return Response({
                "code": 200, "retries_used": [ret1, ret2], "image_data": b64,
                "file": str(res2), "prompt": [p1, p2], "success": [s1, s2]
            })
        except Exception as e:
            return Response({"code": 500, "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PromptPdView(APIView):
    def post(self, request):
        req_data = AskRequestModel(**request.data)
        all_prompt = ask_api.get_final_prompt(fetch_data(), ask_ai_for_pd.get_ask_pd_prompt(req_data))
        return Response({"code": 200, "all_prompt": all_prompt})

class PromptGraphView(APIView):
    def post(self, request):
        req_data = AskRequestModel(**request.data)
        all_prompt = ask_api.get_final_prompt(fetch_data(), ask_ai_for_graph.get_ask_graph_prompt(req_data, llm, tmp_file=True, img_type=False))
        return Response({"code": 200, "all_prompt": all_prompt})

