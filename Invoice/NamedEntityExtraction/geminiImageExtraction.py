from google import genai
import PIL.Image
import os
from dotenv import load_dotenv

load_dotenv(override=True)

image_path_1 = "invoiceSample1.png"
pil_image = PIL.Image.open(image_path_1)

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What does image contain?",
              pil_image])

print(response.text)