import base64
import requests
import os
from mistralai import Mistral
from pydantic import BaseModel, Field
from typing import List

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

# Path to your image
image_path = "invoiceSample1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

# Retrieve the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]

# Specify model
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Define the messages for the chat
messages = [
    {
            "role": "system", 
            "content": "Extract the bill to information only not all invoice stuff into the structured response_format passed as input."
        },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}" 
            }
        ]
    }
]

class Product(BaseModel):
    Product_Name : str
    Product_Descritpion: str
    Quantity: float
    Price: float

class BillTo(BaseModel):
    Customer_Address: str
    Customer_Name : str
    Customer_Phone: str

class CompanyDetails(BaseModel):
    Company_Name : str
    Company_Address : str
    Company_phone: str

class Info(BaseModel):
    BillTo: BillTo
    Company: CompanyDetails
    Products : List[Product]
    Tax: float = Field(description="Set default to 5")
    Total_cost: float = Field(description="Multiply the total cost by tax value and divide by 100 then add this to the original total cost")

info_instance = BillTo(
    Customer_Address="123 Main St",
    Customer_Name="John Doe",
    Customer_Phone="555-1234"
)

# Convert Info instance to dictionary
info_dict = info_instance.model_dump()


# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages,
    response_format= info_dict
)

# Print the content of the response
print(chat_response.choices[0].message.content)