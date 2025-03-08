
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def test():
  import logfire

  api_key= os.getenv("Logfire_Write_Token")
  print(api_key)
  logfire.configure(token=api_key)
  logfire.info("Hello, {name}!", name="Algorithmica")

test()