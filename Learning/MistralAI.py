import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "List 10 possible titles for a fantasy book. Give a list only.",
        },
    ],
    temperature=0,
    frequency_penalty=1.5
)

print(chat_response.choices[0].message.content)