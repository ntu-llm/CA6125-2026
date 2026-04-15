from openai import OpenAI
# Configured by environment variables
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=messages,
    max_tokens=32768,
    temperature=1.0,
    top_p=1.0,
    presence_penalty=2.0,
    extra_body={
        "top_k": 20,
    }, 
)
print("Chat response:", chat_response)
