import sqlite3
import os
import json

def create_metadata_json(src_dir: str, db_file: str, json_file: str):
  db_path = os.path.join(src_dir, db_file)
  json_path = os.path.join(src_dir, "metadata")
  if not os.path.exists(json_path):
        os.makedirs(json_path)
        
  json_file = os.path.join(json_path, json_file)
  with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema = []

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            table_info = {
                "table": table_name,
                "columns": [{"name": col[1], "type": col[2]} for col in columns]
            }

            schema.append(table_info)
            
        with open(json_file, "w") as f:
            json.dump(schema, f, indent=4)
            
if __name__ == "__main__":
    src_dir = os.path.expanduser(
        "~/genAI/Chatapp/DbBasedChatApp/data"
    )
    db_file = "chinook_Sqlite.sqlite"
    json_file = "chinook_schema.json"
    create_metadata_json(src_dir, db_file, json_file)