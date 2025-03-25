import fitz  # PyMuPDF
import json
import os
from pathlib import Path
from fuzzywuzzy import fuzz

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)

    sections = []
    current_section_title = ""
    current_chunk = ""
    # Predefined section titles
    section_titles = set([
        "Work Experience", "Education", "Skills", "Certifications", "Internship",
        "Projects", "Languages", "Awards", "Summary", "Objective", "Contact Details", "Profile"
    ])

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Get text blocks and their metadata (including font size)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Check if the font size is large enough to be considered a section title
                        if "Bold" in span["font"]:
                            # Skip if the text looks like a bullet point (common bullet characters)
                            if span["text"].strip().startswith("•") or span["text"].strip().startswith("-"):
                                continue
                            # If there is an existing section, save the previous section
                            if current_section_title and current_chunk:
                                sections.append({
                                    "source": pdf_path._str,
                                    "sectionTitle": current_section_title,
                                    "chunk": current_chunk.strip()
                                })
                            # Start a new section with the detected title
                            current_section_title = span["text"].strip()
                            
                            # Perform fuzzy matching with the section_titles set
                            matched_title = None
                            highest_ratio = 0
                            for title in section_titles:
                                # Compare using fuzzy matching (Ratio score)
                                ratio = fuzz.partial_ratio(current_section_title.lower(), title.lower())
                                if ratio > highest_ratio:
                                    highest_ratio = ratio
                                    matched_title = title
                                    
                            # If a close match is found, set it as the section title
                            # if matched_title:
                            #     current_section_title = matched_title + " " + current_section_title
                                
                            current_chunk = ""  # Reset current chunk for new section
                        else:
                            # Add the text to the current section's chunk
                            current_chunk += span["text"] + " "

        # After processing the page, add the last section
        if current_section_title and current_chunk:
            sections.append({
                "source": pdf_path._str,
                "sectionTitle": current_section_title,
                "chunk": current_chunk.strip()
            })

    return sections
  
def save_metadata_to_json(metadata, output_json_path):
    # Check if the file exists
    if os.path.exists(output_json_path):
        # If the file exists, read the existing data
        with open(output_json_path, 'r') as json_file:
            existing_metadata = json.load(json_file)
    else:
        # If the file doesn't exist, initialize an empty list
        existing_metadata = []

    # Append the new metadata to the existing data
    existing_metadata.extend(metadata)

    # Save the updated data back to the file
    with open(output_json_path, 'w') as json_file:
        json.dump(existing_metadata, json_file, indent=4)
        
def delete_existing_file(output_json_path):
    # Delete the file if it exists
    if os.path.exists(output_json_path):
        os.remove(output_json_path)
        print(f"{output_json_path} has been deleted.")
    else:
        print(f"{output_json_path} does not exist. No file to delete.")

kb_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/resume")
delete_existing_file('metadata.json')
# Main process
for file in Path(kb_dir).glob("*.pdf"):
  metadata = extract_sections(file)

  # Save metadata as JSON
  output_json_path = "metadata.json"
  save_metadata_to_json(metadata, output_json_path)

print("Metadata has been saved to JSON.")
