import sys
import re
from collections import OrderedDict
import json
import pickle

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def is_chapter_header(line: str) -> bool:
    """
    Simple heuristic to decide if a line looks like a chapter header.
    Adjust this logic to better match your PDF structure.
    """
    line_stripped = line.strip()
    # Condition 1: The line starts with the word "Chapter" or "CHAPTER".
    # Condition 2: The line is in all uppercase and longer than a few characters.
    return (
        line_stripped.lower().startswith("chapter")
        or (line_stripped == line_stripped.upper() and len(line_stripped) > 4)
    )

def is_page_number(line: str) -> bool:
    """
    A naive check to see if the line is purely a digit (likely a page number).
    """
    line_stripped = line.strip()
    return line_stripped.isdigit()

def parse_pdf(pdf_path: str) -> OrderedDict:
    """
    Parse the PDF and return an OrderedDict where:
      keys   -> Chapter (or section) title
      values -> List of text lines in that section.
    """
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        chapter_data = OrderedDict()

        current_chapter = "Introduction"
        chapter_data[current_chapter] = []

        for page_index in range(len(reader.pages)):
            text = reader.pages[page_index].extract_text()
            if not text:
                continue

            for line in text.splitlines():
                # Remove page numbers if detected
                if is_page_number(line):
                    continue

                if is_chapter_header(line):
                    # Start a new chapter section
                    current_chapter = line.strip()
                    chapter_data[current_chapter] = []
                else:
                    # Append this line to the current chapter
                    chapter_data[current_chapter].append(line)

    return chapter_data

def parse_text_file(text_path: str) -> OrderedDict:
    """
    Parse a plain text file and return an OrderedDict where:
        keys   -> Chapter (or section) title
        values -> List of text lines in that section.
    """
    chapter_data = OrderedDict()
    current_chapter = "Introduction"
    chapter_data[current_chapter] = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_chapter_header(line):
                current_chapter = line
                chapter_data[current_chapter] = []
            else:
                chapter_data[current_chapter].append(line)
    return chapter_data

def parse_epub(epub_path: str) -> OrderedDict:
    """
    Parse an EPUB file and return an OrderedDict where:
        keys   -> Chapter (or section) title
        values -> List of text lines in that section.
    """
    book = epub.read_epub(epub_path)
    chapter_data = OrderedDict()
    current_chapter = "Introduction"
    chapter_data[current_chapter] = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Look for chapter indicators in the document
            chapter_found = False
            
            # Check for common chapter heading patterns
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                heading_text = heading.get_text(strip=True)
                if is_chapter_header(heading_text):
                    current_chapter = heading_text
                    chapter_data[current_chapter] = []
                    chapter_found = True
                    
                    # Get the content following this heading until the next heading
                    content = []
                    for elem in heading.find_next_siblings():
                        if elem.name in ['h1', 'h2', 'h3']:
                            break
                        if elem.get_text(strip=True):
                            content.extend(line.strip() for line in elem.get_text().splitlines() if line.strip())
                    chapter_data[current_chapter].extend(content)
            
            # If no chapter heading was found, check if the content itself indicates a chapter
            if not chapter_found:
                text_content = soup.get_text(strip=True)
                lines = [line.strip() for line in text_content.splitlines() if line.strip()]
                
                for line in lines:
                    if is_chapter_header(line):
                        current_chapter = line
                        chapter_data[current_chapter] = []
                        chapter_found = True
                    else:
                        chapter_data[current_chapter].append(line)

    # Clean up empty chapters
    return OrderedDict((k, v) for k, v in chapter_data.items() if v)

def main():
    if len(sys.argv) < 2:
        print("Usage: python readPDF.py <path to file>")
        sys.exit(1)
    file_path = sys.argv[1]
    output_path = sys.argv[2]

    if file_path.lower().endswith('.pdf'):
        structured_text = parse_pdf(file_path)
    elif file_path.lower().endswith('.epub'):
        structured_text = parse_epub(file_path)
    else:
        structured_text = parse_text_file(file_path)

    # Convert the ordered dictionary to a list of objects
    structured_array = []
    for chapter_title, lines in structured_text.items():
        structured_array.append({
            "chapter_title": chapter_title,
            "chapter_content": lines
        })

    # Write out the data to a pickle file instead of JSON
    output_path = output_path.replace('.json', '.pkl')  # Change extension
    with open(output_path, "wb") as outfile:  # Note: "wb" for binary write
        pickle.dump(structured_array, outfile)

    # Print the extracted structure
    for chapter_title, lines in structured_text.items():
        print(f"\n=== {chapter_title} ===")
        for line in lines:
            print(line)

if __name__ == "__main__":
    main()
