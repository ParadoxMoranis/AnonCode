import os
import requests
import json
import base64
import time
from typing import List, Dict, Union

class APIClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, messages: List[Dict], temperature: float) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        model_name = (self.model or "").lower()
        # Bailian Qwen3 non-streaming calls require explicitly disabling thinking.
        if "qwen3" in model_name:
            payload["enable_thinking"] = False

        return payload

    def encode_image(self, image_path: str) -> str:
        """将图片文件转换为 Base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat_completion(self, messages: List[Dict], temperature: float = 0.2, max_retries: int = 3) -> str:
        """
        发送请求到 OpenAI 兼容接口
        """
        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(messages, temperature)

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                return content
            except Exception as e:
                print(f"⚠️ [API Warning] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1)) # 指数退避
                else:
                    print(f"❌ [API Error] Request failed after retries.")
                    return ""
        return ""

    def parse_json_response(self, text: str) -> Dict:
        """鲁棒的 JSON 解析，处理 ```json 标记"""
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print(f"❌ [JSON Error] Failed to parse: {text}")
            return {}
