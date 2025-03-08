import duckdb

# Create an in-memory DuckDB database
con = duckdb.connect(database=':memory:')

# Create a table
def CreateTable(createStatement: str):
  con.execute(createStatement)

# Insert some data
def SaveData(insertStatement: str):
  con.execute(insertStatement)

# Query the data
def GetData(selectStatement: str):
  results = con.execute(selectStatement).fetchall()
  return results