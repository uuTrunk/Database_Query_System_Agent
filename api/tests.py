from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock
from api.views import config_data
from pathlib import Path

class APITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.login_url = reverse('login')
        self.ask_pd_url = reverse('ask_pd')
        self.ask_graph_url = reverse('ask_graph')
        
        config_data["auth"] = {
            "username": "admin",
            "password": "password123"
        }

    def test_login_success(self):
        response = self.client.post(self.login_url, {"username": "admin", "password": "password123"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
    def test_login_failure(self):
        response = self.client.post(self.login_url, {"username": "admin", "password": "wrongpassword"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
    @patch('api.views.fetch_data')
    @patch('ask_ai.ask_ai_for_pd.ask_pd')
    def test_ask_pd_success(self, mock_ask_pd, mock_fetch_data):
        mock_fetch_data.return_value = ["dummy", "dummy", "dummy"]
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"answer": "test_answer"}
        mock_ask_pd.return_value = (mock_result, 1, "test_prompt", 1.0)
        
        response = self.client.post(self.ask_pd_url, {"question": "What is the total?", "concurrent": 1, "retries": 1}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    @patch('api.views.fetch_data')
    @patch('ask_ai.ask_ai_for_pd.ask_pd')
    def test_ask_pd_failure(self, mock_ask_pd, mock_fetch_data):
        mock_fetch_data.return_value = ["dummy", "dummy", "dummy"]
        mock_ask_pd.return_value = (None, 2, "failed_prompt", 0.0)
        
        response = self.client.post(self.ask_pd_url, {"question": "Fail question"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)
        self.assertEqual(response.data.get("code"), 504)

    @patch('api.views.fetch_data')
    @patch('ask_ai.ask_ai_for_graph.ask_graph')
    @patch('pathlib.Path.read_bytes')
    def test_ask_graph_success(self, mock_read_bytes, mock_ask_graph, mock_fetch_data):
        mock_fetch_data.return_value = ["dummy", "dummy", "dummy"]
        mock_ask_graph.return_value = ("dummy_img_path", 1, "test_prompt", 1.0)
        mock_read_bytes.return_value = b'fake_image_bytes'
        
        response = self.client.post(self.ask_graph_url, {"question": "Draw graph"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_invalid_payload_pydantic_error(self):
        response = self.client.post(self.ask_pd_url, {"concurrent": 2}, format="json")
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    @patch('api.views.fetch_data')
    @patch('ask_ai.ask_ai_for_pd.ask_pd')
    @patch('ask_ai.ask_ai_for_graph.ask_graph')
    @patch('pathlib.Path.read_bytes')
    def test_ask_graph_steps_success(self, mock_read_bytes, mock_ask_graph, mock_ask_pd, mock_fetch_data):
        mock_fetch_data.return_value = ["dummy", "dummy", "dummy"]
        
        mock_pd_result = MagicMock()
        mock_pd_result.to_dict.return_value = {"answer": "step1"}
        mock_ask_pd.return_value = (mock_pd_result, 1, "prompt1", 1.0)
        
        mock_ask_graph.return_value = ("dummy_img_path", 1, "prompt2", 1.0)
        mock_read_bytes.return_value = b'fake_image_bytes'
        
        url = reverse('ask_graph_steps')
        response = self.client.post(url, {
            "question": "step1",
            "concurrent": [1, 1],
            "retries": [0, 0]
        }, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

