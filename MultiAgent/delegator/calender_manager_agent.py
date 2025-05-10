from pydantic_ai.models.gemini import GeminiModel
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from dummy_calendar import DummyCalendar

load_dotenv(override=True)
logfire.instrument_pydantic_ai()

calendar = DummyCalendar()

def add_event_tool(title: str, start_time: str, end_time: str):
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    event = calendar.add_event(title, start_dt, end_dt)
    if event:
        return f"Event '{title}' added from {start_time} to {end_time}."
    else:
        alt = calendar.suggest_alternate_time((end_dt - start_dt).seconds // 60)
        if alt:
            return f"Time slot is busy. Suggested alternate: {alt[0]} - {alt[1]}"
        else:
            return "No available time slot in the next 24 hours."

def update_event_tool(title: str, new_start_time: str, new_end_time: str):
    new_start_dt = datetime.fromisoformat(new_start_time)
    new_end_dt = datetime.fromisoformat(new_end_time)
    event = calendar.update_event(title, new_start_dt, new_end_dt)
    return f"Event '{title}' updated." if event else "Unable to update event. Conflict or not found."

def delete_event_tool(title: str):
    result = calendar.delete_event(title)
    return f"Event '{title}' deleted." if result else "Event not found."

def suggest_time_tool(duration_minutes: int = 60):
    alt = calendar.suggest_alternate_time(duration_minutes)
    if alt:
        return f"Suggested time: {alt[0]} - {alt[1]}"
    return "No available time slot found."


calendar_agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp",provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
            system_prompt= """
                  You are calendar managing agent.
                  Given an event i.e. event title and date time, either add it to the calendar, if the calendar looks free
                  else, return another suggestion on time.
                  """,
            tools=[add_event_tool,
                   update_event_tool,
                   delete_event_tool,
                   suggest_time_tool
        ]
      )

# test_prompts = [
#     "Schedule a design review on April 26th 2025 from 2 PM to 3 PM.",
#     "Book a follow-up meeting on April 26th 2025 from 2:30 PM to 3:30 PM.",
#     "Reschedule the design review to April 26th 2025 at 4 PM.",
#     "Cancel the follow-up meeting.",
#     "Suggest a 45-minute free slot for a meeting tomorrow."
# ]

# for prompt in test_prompts:
#   response = calendar_agent.run_sync(user_prompt=prompt)
#   print(response.data)

