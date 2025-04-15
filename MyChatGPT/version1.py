from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
import gradio as gr
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.providers.google_gla import GoogleGLAProvider

load_dotenv(override=True)
logfire.instrument_pydantic_ai()

chatAgent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp",provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
            system_prompt= """
                  You are a helpful assistant. Get information if you have it, else acknowledge you don't have data
                """
      )

def convert_gradio_history_to_pydantic_ai(history):
    history_pydantic_ai_format = []
    for msg in history[:10]:
        if msg["role"] == "user":
            tmp = ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
        elif msg["role"] == "assistant":
            tmp = ModelResponse(parts=[TextPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
    return history_pydantic_ai_format


async def respond(message, history):
    history_pydantic_ai_format = convert_gradio_history_to_pydantic_ai(history)
    
    async with chatAgent.run_stream(message, message_history=history_pydantic_ai_format) as result:
        async for chunk in result.stream_text():
            yield chunk


chatInterface = gr.ChatInterface(
    respond,
    type="messages",
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your question"),
    title="MyChatGPT",
    description="Ask any question and get the answer."
)

chatInterface.launch()