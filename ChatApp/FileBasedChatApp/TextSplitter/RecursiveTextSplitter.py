from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0, length_function= len)

def split_text(text: str):
  chunks = text_splitter.split_text(text=text)
  print(len(chunks))
  for chunk in chunks:
    print(len(chunk), chunk)
  return chunks

if __name__ == "__main__":
  split_text("""What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. 
I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. 
They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called 
"data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the 
basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini 
Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — 
sitting up on a raised floor under bright fluorescent lights.
""")