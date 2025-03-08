from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("Open_API_Key"))


def chat_with_openai(query):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


# print(chat_with_openai("What is recursion in programming."))
print(chat_with_openai("how to use OPENAI class while coding."))