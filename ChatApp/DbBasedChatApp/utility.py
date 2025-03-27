import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.inmemmDB import CreateTable, GetData, SaveData

CreateTable('CREATE TABLE expenses (id INTEGER PRIMARY KEY, store_name TEXT, category TEXT, amount FLOAT)')
class Expense:
    Id: int
    StoreName: str
    Category: str
    Amount: float
    
    def __init__(self, id, storeName, category, amount):
        self.Id = id
        self.Category = category
        self.StoreName = storeName
        self.Amount = amount

def save_expense(expense: Expense):
        insertStmmt = f"INSERT INTO expenses VALUES ({expense.Id}, '{expense.StoreName}', '{expense.Category}' , {expense.Amount})"
        print(insertStmmt)
        SaveData(insertStatement=insertStmmt)
        
save_expense(Expense( 1, "Dm", "Groceries",100.0))
save_expense(Expense( 2, "Ikea", "Furnitures",1000.0))


def get_expenses(query: str):
    return GetData(query)