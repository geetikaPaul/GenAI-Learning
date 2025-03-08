import uuid
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.gemini import GeminiModel
import os
import logfire
from typing import Optional
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.AgentBuilder import get_agent, get_prompt
from Utils.inmemmDB import CreateTable, GetData, SaveData
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic_ai import RunContext, Agent, Tool
from dataclasses import dataclass

logfire.configure(token=os.getenv("Logfire_Write_Token"))

file_path = os.path.expanduser("~/genAI/Invoice_NamedEntityExtraction/data/IkeaBill.jpeg") #input("Please enter the file path: ")

class Expenses(BaseModel):
    Id: int = Field(description="create random integer")
    StoreName: Optional[str] = Field(description= "Pick the name on the top of the bill")
    Category: str = Field(description="if items are grocery items then grocery, electronics as electronic and so on")
    Amount: float
    
@dataclass
class SystemDeps:
    userName: str

agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= "Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool. query DB to see if samme category already exists. Save to DB",
            result_type = Expenses
      )

dbAgent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= "query DB to see if given Category already exists. Yes/ No. If yes, Print StoreName and Amount from the DB for matching Category. Tell all categories that are in dB, if you can't find answer to previous question.",
            deps_type= SystemDeps
      )

# get_agent(system_prompt="Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool.", 
#                   #result_type=Expenses, 
#                   model_name="gemini",
#                   deps_type = SystemDeps)
#agent = get_agent(file_path = file_path, system_prompt="Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool.", result_type=Expenses, model_name="gemini")

image_prompt = get_prompt(user_prompt="Extract total bill amount and category of bill", file_path=os.path.expanduser
                                 (file_path))

@agent.system_prompt
def add_name(ctx: RunContext[SystemDeps]) -> str :
    return f"With every response add username: {ctx.deps.userName}"

    
CreateTable('CREATE TABLE expenses (id INTEGER PRIMARY KEY, store_name TEXT, category TEXT, amount FLOAT)')
                                
@dbAgent.tool_plain
def get_expenses():
    # db = SessionLocal()
    # expenses = db.query(ExpenseDB).all()
    # db.close()
    # return expenses
    
    return GetData('SELECT * FROM expenses')
    # return Expenses(StoreName="Ikea", Category="Other",Amount=100.0, AlreadyCategoryExists=False)
    
@agent.tool_plain                          
def save_expense(expense: Expenses):
        insertStmmt = f"INSERT INTO expenses VALUES ({expense.Id}, '{expense.StoreName}', '{expense.Category}' , {expense.Amount})"
        print(insertStmmt)
        SaveData(insertStatement=insertStmmt)
    
@agent.tool_plain
def get_category(store_name: str) -> str:
    """Returns the category of the store"""
    if "Edeka" in store_name:
        return "Grocery"
    elif "MediaMarkt" in store_name:
        return "Electronics"
    elif "H&M" in store_name:
        return "Clothing"
    elif "Ikea" in store_name:
        return "Household"
    else:
        return "Other"
    
systemDeps = SystemDeps(userName="GP")

user_prompt = get_prompt("Find category of the shop", file_path=file_path)
save_expense(Expenses(Id = 1, StoreName="Dm", Category="Other",Amount=100.0))

def callAgent():
    result = agent.run_sync(user_prompt=user_prompt, 
                        deps=systemDeps)
    invoice = result.data
    #save_expense(invoice)
    print(invoice)
    return invoice.Category


autonomousAgent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= "Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool. query DB to see if samme category already exists. Save to DB. query DB to see if given Category already exists. Yes/ No. If yes, Print StoreName and Amount from the DB for matching Category. Tell all categories that are in dB, if you can't find answer to previous question.",
            deps_type= SystemDeps,
            tools=[Tool(save_expense, takes_ctx=False), Tool(add_name, takes_ctx=True),Tool(get_expenses, takes_ctx=False), ]
      )
            
with logfire.span("Calling model") as span:
    # category = callAgent()
    # #category = "Other"
    # result = dbAgent.run_sync(f"Tell me if {category} exists in DB: Yes/ No", deps = systemDeps)
    result = autonomousAgent.run_sync(user_prompt, deps=systemDeps)
    print(result.data)

      