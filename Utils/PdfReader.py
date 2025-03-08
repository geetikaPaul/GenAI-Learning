from langchain_community.document_loaders.pdf import PyPDFLoader

def get_text_from_pdf(path):
    pdf_loader = PyPDFLoader(path)
    docs = pdf_loader.load()
    return docs

def get_pdf_first_page(path):
    docs = get_text_from_pdf(path=path)
    return docs[0].page_content