from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0, length_function= len, separator="")

def split_text(text: str):
  chunks = text_splitter.split_text(text=text)
  print(len(chunks))
  for chunk in chunks:
    print(len(chunk), chunk)
  return chunks

if __name__ == "__main__":
  split_text("The generative ai technology showcases the potential of AI for businesses, individuals and society.")