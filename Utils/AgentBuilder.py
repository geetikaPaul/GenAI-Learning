
from dotenv import load_dotenv
import os
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.gemini import GeminiModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.PdfReader import get_pdf_first_page
from Utils.ImagePrompt import ImageLoaderBase64
from pathlib import Path

load_dotenv(override=True)

def get_model(model_name: str = None):
  if model_name == "gemini":
    return GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              )
  elif model_name == "pixtral":
      return MistralModel(
                model_name="pixtral-12b-2409", api_key=os.environ["MISTRAL_API_KEY"]
                )
  else:
    return MistralModel(
                model_name="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"]
                )

def get_agent(file_path: str, system_prompt: str, result_type: type, model_name: str = None, deps_type: type = None):
    if isImgFile(file_path):
      return Agent(model = get_model("pixtral"),
            system_prompt= system_prompt,
            result_type= result_type,
            deps_type= deps_type
      )
    elif isPdfFile(file_path):
      return Agent(
            model=get_model("gemini"),
            system_prompt= system_prompt,
            result_type= result_type,
            deps_type= deps_type
      )
    elif model_name:
      return Agent(
            model=get_model(model_name),
            system_prompt= system_prompt,
            result_type= result_type,
            deps_type= deps_type
      )
    else:
      return Agent(
            model=get_model("mistral"),
            system_prompt= system_prompt,
            result_type= result_type,
            deps_type= deps_type
      )
      
def get_agent(system_prompt: str, result_type : type = None, model_name: str = None, deps_type: type = None):
    params = {
        "model": get_model(),
        "system_prompt": system_prompt,
    }
    if model_name is not None:
        params["model"] = get_model(model_name)
    if result_type is not None:
        params["result_type"] = result_type
    if deps_type is not None:
        params["deps_type"] = deps_type
        
    return Agent(**params)
    
def get_prompt(user_prompt: str = None, file_path: str = None):
  if file_path and isImgFile(file_path):
    # obsolete way: pydantic ai supports image input now
    # return ImageLoaderBase64(user_prompt=user_prompt, image_file_path=file_path).encoded_message_with_image
    image_extension = file_path.split('.')[-1].lower()
    if image_extension not in ['jpeg', 'jpg', 'png']:
        raise ValueError("Unsupported image format. Please use JPEG or PNG.")

    mime_type = "image/jpeg" if image_extension in ['jpeg', 'jpg'] else "image/png"
    return [user_prompt, BinaryContent( data = Path(file_path).read_bytes(), media_type = mime_type)]
  if file_path and isAudFile(file_path):
    aud_extension = file_path.split('.')[-1].lower()
    if aud_extension not in ['mp3']:
        raise ValueError("Unsupported audio format. Please use mp3.")

    media_type = "audio/mpeg"
    return [user_prompt, BinaryContent( data = Path(file_path).read_bytes(), media_type = media_type)]
  if file_path and isPdfFile(file_path):
    return get_pdf_first_page(os.path.expanduser(file_path))
  else:
    return user_prompt
    
def isImgFile(file_path: str):
    return file_path.lower().endswith('.jpeg') or file_path.lower().endswith('.png')
  
def isAudFile(file_path: str):
  return file_path.lower().endswith('.mp3')
  
def isPdfFile(file_path: str):
    return file_path.lower().endswith('.pdf')