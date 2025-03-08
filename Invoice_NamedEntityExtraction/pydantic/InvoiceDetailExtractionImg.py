from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import logfire
from typing import List
import base64
from google.genai import types
import pathlib
import PIL.Image
from io import BytesIO
from pydantic_ai.models.mistral import MistralModel
from typing import Optional

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

class Product(BaseModel):
    Product_Name : str = Field(description= "Item name & description")
    Quantity: float
    Price: float = Field(description= "Rate")
    Amount: float

class BillTo(BaseModel):
    Address: str
    Name : str
    Phone: str

class ShipTo(BaseModel):
    Address: str
    Name : str
    Phone: str

class SellerDetails(BaseModel):
    Name : str = Field(description="name of the selling company")
    Address : str = Field(description="address of the selling company")
    phone: str

class Info(BaseModel):
    Bill_to: BillTo
    Ship_to: Optional[ShipTo]
    Seller: SellerDetails
    Products : List[Product]
    Tax_Rate: float = Field(description="tax rate in percentage")
    Total_cost: float

    def set_ship_to_as_bill_to(self):
        """
        Set the ShipTo details as the BillTo details if ShipTo is missing.
        """
        if not self.Ship_to:  # If ship_to is None or empty
            self.Ship_to = self.Bill_to

agent = Agent(
    model=MistralModel(
        model_name="pixtral-12b-2409", api_key=os.environ["MISTRAL_API_KEY"]
        )
    ,
    #system_prompt="convert and replace fields with value NA or empty as 'hahaaa.",
    result_type=Info
)

class ImageLoaderBase64:

    def __init__(self, user_prompt: str, image_file_path: str):
        with open(image_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        self.encoded_message_with_image = [
            {"type": "text", "text": f"{user_prompt}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]

image_prompt = ImageLoaderBase64(user_prompt="Extract bill details", image_file_path=os.path.expanduser
                                 ("~/genAI/Invoice_NamedEntityExtraction/data/invoice.png"))

# Gemini trial
# image_path = "invoiceSample1.png"
# image = PIL.Image.open(image_path)

# def image_to_base64(image_path: str) -> str:
#     with open(image_path, "rb") as img_file:
#         # Convert image to base64 encoding
#         encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
#     return encoded_image
# encoded_image = image_to_base64(image_path)
# messages = ["What's in this image?", encoded_image]

# message = {
#   "contents": ["What is in this image?",encoded_image]
# }


#with logfire.span("Calling model") as span:
result = agent.run_sync(image_prompt.encoded_message_with_image) 
invoice = result.data
invoice.set_ship_to_as_bill_to()
print(invoice)