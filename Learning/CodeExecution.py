from google import genai
from google.genai import types
import os

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))

response = client.models.generate_content(
  model='gemini-2.0-flash-exp',
  contents='What is the sum of the first 50 prime numbers? '
           'Generate and run C# code for the calculation, and make sure you get all 50.',
  config=types.GenerateContentConfig(
    tools=[types.Tool(
      code_execution=types.ToolCodeExecution
    )]
  )
)

def display_code_execution_result(response):
  for part in response.candidates[0].content.parts:
    if part.text is not None:
      print(part.text)
    if part.executable_code is not None:
      code_html = f'<pre style="background-color: green;">{part.executable_code.code}</pre>' # Change code color
      print(code_html)
    if part.code_execution_result is not None:
      print(part.code_execution_result.output)
    print("---")

display_code_execution_result(response)