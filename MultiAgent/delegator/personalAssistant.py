from pydantic_ai.models.gemini import GeminiModel
from typing import Any
from pydantic_ai import RunContext
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from calender_manager_agent import *
from notes_manager_agent import *

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

personal_assistant_agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp",provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
            system_prompt= """
                  You are a personal assistant.
                  You can help me book calendar appointments
                  You can help me take notes when asked to take a note of something
                  """
      )

@personal_assistant_agent.tool()
async def call_calendar_manager_agent(ctx: RunContext, prompt: str) -> Any:
    result = await calendar_agent.run(prompt)
    return result.data

@personal_assistant_agent.tool()
async def call_note_manager_agent(ctx: RunContext, prompt: str) -> Any:
    result = await note_agent.run(prompt)
    return result.data

test_prompts = [
    "Schedule a design review on April 26th 2025 from 2 PM to 3 PM.",
    #"Book a follow-up meeting on April 26th 2025 from 2:30 PM to 3:30 PM.",
    "Take a note of - I have an appointment at doctor tomorrow",
    "Note that Doctor appointment updated to Monday morning"
    # "Reschedule the design review to April 26th 2025 at 4 PM.",
    # "Cancel the follow-up meeting.",
    # "Suggest a 45-minute free slot for a meeting tomorrow."
]

for prompt in test_prompts:
    logfire.instrument_pydantic_ai()
    response = personal_assistant_agent.run_sync(user_prompt=prompt)
    print(response.data)

