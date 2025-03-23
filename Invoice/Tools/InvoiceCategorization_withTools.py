import uuid
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logfire
from typing import Optional
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.AgentBuilder import get_agent, get_prompt
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logfire.configure(token=os.getenv("Logfire_Write_Token"))

file_path = "~/genAI/Invoice/data/IkeaBill.jpeg" #input("Please enter the file path: ")

class Expenses(BaseModel):
    StoreName: Optional[str] = Field(description= "Pick the name on the top of the bill")
    Category: str = Field(description="if items are grocery items then grocery, electronics as electronic and so on")
    Amount: float

#agent = get_agent(file_path = file_path, system_prompt="Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool.", result_type=Expenses, model_name="gemini")

image_prompt = get_prompt(user_prompt="Extract total bill amount and category of bill", file_path=os.path.expanduser
                                 (file_path))

# Create an in-memory SQLite database
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
                                
# Define the Expense model
class ExpenseDB(Base):
    __tablename__ = "expenses"
    id = Column(String, primary_key=True, index=True)
    store_name = Column(String, index=True)
    category = Column(String, index=True)
    amount = Column(Float, index=True)
                                
# Create the expenses table
Base.metadata.create_all(bind=engine)

def get_expenses():
    db = SessionLocal()
    expenses = db.query(ExpenseDB).all()
    db.close()
    return expenses

# Fetch and print all expenses
print("Printing data before prompt")
all_expenses = get_expenses()
for expense in all_expenses:
    print(f"ID: {expense.id}, Store Name: {expense.store_name}, Category: {expense.category}, Amount: {expense.amount}")
      
# @agent.tool_plain                          
def save_expense(expense: Expenses):
        """Saves the expense in the database"""
        db = SessionLocal()
        db_expense = ExpenseDB(
        id=str(uuid.uuid4()),
        store_name=expense.StoreName,
        category=expense.Category,
        amount=expense.Amount
        )
        db.add(db_expense)
        db.commit()
        db.refresh(db_expense)
        db.close()
        return db_expense

agent = get_agent(system_prompt="Extract bill data. First, attempt to find category based on your knowledge. If you don't know the answer, retrieve category through an external tool.", result_type=Expenses, model_name="gemini")
  
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
    

with logfire.span("Calling model") as span:
    result = agent.run_sync("I bought stuff from Tesco for 100 euros")
    invoice = result.data
    save_expense(invoice)
    print(invoice)

print("Printing data after prompt")
all_expenses = get_expenses()
for expense in all_expenses:
    print(f"ID: {expense.id}, Store Name: {expense.store_name}, Category: {expense.category}, Amount: {expense.amount}")
      