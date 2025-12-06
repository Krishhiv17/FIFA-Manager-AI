# src/llm_client.py

import os
import requests
from requests.exceptions import RequestException
from typing import List, Dict
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env (HF_TOKEN etc.)
load_dotenv()

# Hugging Face Router endpoint (OpenAI-compatible)
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Your HF token (must be set in .env or environment)
HF_TOKEN = os.getenv("HF_TOKEN")

# The model you want to use
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN is not set. Please add it to your environment or .env file."
    )

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


def call_llama(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Call Llama 3 8B Instruct via Hugging Face Router using an
    OpenAI-compatible /v1/chat/completions interface.

    Parameters
    ----------
    messages : list of {"role": "system"|"user"|"assistant", "content": str}
        Conversation so far.
    max_tokens : int
        Maximum number of new tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling parameter.

    Returns
    -------
    str
        The assistant's message content.
    """
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
    except RequestException as exc:
        raise RuntimeError(
            "LLM request failed (network/endpoint issue). "
            "Check internet connectivity and HF_TOKEN."
        ) from exc
    data = response.json()

    # Expected structure:
    # {
    #   "choices": [
    #     {
    #       "index": 0,
    #       "message": {"role": "assistant", "content": "..."},
    #       ...
    #     }
    #   ],
    #   ...
    # }
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: return raw JSON if format is unexpected
        return str(data)


if __name__ == "__main__":
    # Quick smoke test: run `python src/llm_client.py`
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi in one short sentence."},
    ]
    print(call_llama(test_messages))
