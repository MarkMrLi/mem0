import requests

BASE_URL = "http://127.0.0.1:7000"

# 添加记忆
data = {
    "messages": [
        {"role": "user", "content": "I'm a vegetarian and allergic to nuts."},
        {"role": "assistant", "content": "Got it! I'll remember your dietary preferences."}
    ],
    "user_id": "user123"
}
response = requests.post(f"{BASE_URL}/memories", json=data)
print("Add Result:", response.json())

# 搜索记忆
search_data = {
    "query": "What are my dietary restrictions?",
    "user_id": "user123"
}
response = requests.post(f"{BASE_URL}/search", json=search_data)
print("Search Result:", response.json())