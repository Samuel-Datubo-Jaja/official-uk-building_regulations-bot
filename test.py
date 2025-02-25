import requests

api_key = "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"
url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello, are you working?"}]
}

response = requests.post(url, headers=headers, json=data)
print(f"Status code: {response.status_code}")
print(response.json())