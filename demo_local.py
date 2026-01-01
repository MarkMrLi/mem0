import time
import psutil
import GPUtil
import json
from mem0 import Memory

# ---- 配置 (与你原来一致) ----
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"embedding_model_dims": 768},
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}


# ---- 初始化 Memory ----
m = Memory.from_config(config)

# ---- CRUD 测试 ----
m.add("I am an engineer", user_id="u1")

memories = m.search("What do you know about me?", user_id="u1")
print(memories)
id = memories["results"][0]["id"]
print("-------------------------------------------")
m.update(memory_id=id, data = "I am an software engineer")
memories = m.search("What do you know about me?", user_id="u1")
print(memories)
print("-------------------------------------------")

id = memories["results"][0]["id"]
m.delete(id)

m.add("I like playing basketball", user_id="u1")
memories = m.search("What do you know about me?", user_id="u1")
print(memories)
print("-------------------------------------------")
