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

class Note:
    def __init__(self, title, content, updatedOn):
        self.title = title
        self.content = content
        self.lastUpdatedOn = updatedOn

    def __repr__(self):
        return f"{self.title}: {self.content} - {self.lastUpdatedOn}"
    
class NoteMgr:
    def __init__(self):
        self.notes = []

    def add_note(self, title, content, updatedOn):
        for note in self.notes:
            if note.title == title:
                return self.update_note(self, title, content, updatedOn)
        note = Note(title, content, updatedOn)
        self.notes.append(note)
        return f"Note '{title}' added."

    def update_note(self, title, content, updatedOn):
        for note in self.notes:
            if note.title == title:
                note.content = content
                note.end_time = updatedOn
                return f"Note '{title}' updated."
        return self.add_note(title, content, updatedOn)

    def delete_note(self, title):
        for i, note in enumerate(self.notes):
            if note.title == title:
                del self.notes[i]
                return f"Note '{title}' deleted."
        return "Note not found."

    def list_notes(self):
        return sorted(self.notes, key=lambda x: x.content)
    
noteMrg = NoteMgr()

def add_note_tool(title: str, content: str):
    noteMrg.add_note(title, content, datetime.now)

def update_note_tool(title: str, content: str):
    noteMrg.update_note(title, content, datetime.now)
    
def delete_note_tool(title: str):
    noteMrg.delete_note(title)
    
def list_notes_tool():
    noteMrg.list_notes()

note_agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp",provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
            system_prompt= """
                  You are notes managing agent.
                  Given a new prompt (content), find a good brief title for it, if you find anything very similar to generated title in existing notes then update note else, add it to the notes using tools
                  When asked to delete a note, find the closest note from the list of notes and delete
                  """,
            tools=[add_note_tool,
                   update_note_tool,
                   delete_note_tool,
                   list_notes_tool
                ]
      )

# test_prompts = [
#     "I have an appointment at doctor tomorrow",
#     "my account Id for Gmail: test@gamil.com",
#     "Doctor appointment updated to Monday morning"
# ]

# for prompt in test_prompts:
#   response = note_agent.run_sync(user_prompt=prompt)
#   print(response.data)

