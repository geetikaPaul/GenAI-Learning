import pdfplumber
import fitz  # PyMuPDF for handling PDF metadata and tags
import spacy
import re
import os

# Load spaCy model for text processing
nlp = spacy.load('en_core_web_sm')

# Function to extract text from a PDF using pdfplumber
def extract_pdf_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# Function to extract PDF metadata (title, author, etc.)
def extract_pdf_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    return metadata

# Function to extract PDF bookmarks (tags or outline) for chunk titles
def extract_pdf_bookmarks(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)  # Get the Table of Contents
    if not toc:
        print("No bookmarks found in this PDF.")
    return toc

# Function to detect headings (customized for resumes)
def detect_resume_sections(text):
    headings = []
    # Custom regex patterns for common resume sections (adjust as per your resume structure)
    section_patterns = [
        r'\b(Experience|Work\s*History|Employment)\b',
        r'\b(Education|Academic\s*Background)\b',
        r'\b(Skills|Technical\s*Skills)\b',
        r'\b(Projects|Project\s*Experience)\b',
        r'\b(Certifications|Awards)\b',
        r'\b(Summary|Profile)\b',
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            headings.append(match.group().strip())
    
    return headings

# Function to chunk the content based on detected sections
def chunk_text(text, headings):
    # Split text into chunks using headings as delimiters
    chunks = []
    start_idx = 0
    for heading in headings:
        # Find the position of the heading
        heading_idx = text.find(heading, start_idx)
        if heading_idx != -1:
            if start_idx < heading_idx:
                chunks.append(text[start_idx:heading_idx].strip())
            start_idx = heading_idx
    if start_idx < len(text):
        chunks.append(text[start_idx:].strip())
    return chunks

# Main function to process the PDF and perform content-aware chunking
def process_pdf(pdf_path):
    # Step 1: Extract text from the PDF
    text = extract_pdf_text(pdf_path)

    # Step 2: Extract metadata (title, author, etc.)
    metadata = extract_pdf_metadata(pdf_path)
    title = metadata.get('title', 'Untitled Document')
    author = metadata.get('author', 'Unknown Author')

    # Step 3: Extract bookmarks (tags or outline) for section titles
    bookmarks = extract_pdf_bookmarks(pdf_path)
    bookmark_titles = [bookmark[1] for bookmark in bookmarks]

    if not bookmarks:  # If no bookmarks are found, use regex to detect headings
        print("No bookmarks found, falling back to regex-based headings.")
        bookmark_titles = detect_resume_sections(text)

    # Step 4: Chunk the text based on headings (sections)
    chunks = chunk_text(text, bookmark_titles)

    # Step 5: Combine metadata, bookmarks, and chunked text
    output = {
        'title': title,
        'author': author,
        'bookmarks': bookmark_titles,
        'chunks': []
    }

    for idx, chunk in enumerate(chunks):
        chunk_title = bookmark_titles[idx] if idx < len(bookmark_titles) else f'Chunk {idx + 1}'
        output['chunks'].append({
            'section_title': chunk_title,
            'content': chunk
        })

    return output

# Example Usage
pdf_path = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/resume/JohnDoe.pdf")  # Path to your PDF file
result = process_pdf(pdf_path)

# Print out the title, author, bookmarks, and chunks
print("Title:", result['title'])
print("Author:", result['author'])
print("Bookmarks:", result['bookmarks'])
for chunk in result['chunks']:
    print(f"\nSection: {chunk['section_title']}")
    print(chunk['content'][:300])  # Show a preview of each chunk's content
