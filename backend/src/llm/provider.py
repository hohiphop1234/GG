from __future__ import annotations

import os
import sys
import json
import logging
import requests

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import LLAMA_SERVER_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

class QwenMedicalLLM:
    """
    Class quản lý và gọi mô hình Qwen3-4B thông qua llama-server.
    Cung cấp API gọi LLM trực tiếp chạy siêu tốc cục bộ mà không qua LangChain.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenMedicalLLM, cls).__new__(cls)
            cls._instance.llama_url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
            cls._instance.model_name = "qwen3-4b-thinking"
        return cls._instance

    def generate_answer(self, question: str, max_new_tokens: int = 1024, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = "You are a medical question answering assistant. Answer clearly, cautiously, and remind users to consult healthcare professionals."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
            "frequency_penalty": 1.15
        }
        
        try:
            response = requests.post(self.llama_url, json=payload, timeout=600)
            if response.status_code != 200:
                return f"Lỗi gọi llama-server API: {response.text}"
                
            data = response.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Xử lý dọn dẹp các luồng suy nghĩ <think> nếu mô hình sinh ra
            if "</think>" in answer:
                parts = answer.split("</think>")
                if parts[-1].strip():
                    answer = parts[-1].strip()
                else:
                    answer = parts[0].replace("<think>", "").strip()
            elif answer.startswith("<think>"):
                answer = answer.replace("<think>", "").strip()
                
            return answer.strip() or "Xin lỗi, mô hình AI không tạo được câu trả lời."
            
        except requests.exceptions.ConnectionError:
            return "Lỗi: Không thể kết nối tới llama-server. Vui lòng đảm bảo llama-server đang chạy."
        except Exception as e:
            return f"Exception: {e}"
            
    def stream_answer(self, question: str, system_prompt: str = "", max_new_tokens: int = 2048):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_new_tokens,
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
            "frequency_penalty": 1.15
        }
        
        try:
            response = requests.post(self.llama_url, json=payload, stream=True, timeout=600)
            if response.status_code != 200:
                yield f"Lỗi gọi llama-server API: {response.text}"
                return
                
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(json_str)
                            chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n[Lỗi kết nối llama-server: {e}]"

def get_llm(**kwargs) -> QwenMedicalLLM:
    return QwenMedicalLLM()

def get_streaming_llm(**kwargs) -> QwenMedicalLLM:
    return QwenMedicalLLM()
