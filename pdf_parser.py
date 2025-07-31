import io
import re
from PyPDF2 import PdfReader
import requests

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Extract text from first and last 10 pages of PDF"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, timeout=10, headers=headers)
        response.raise_for_status()
        
        with io.BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)
            # Get first 10 and last 10 pages (with overlap handling)
            target_pages = list(range(0, min(10, total_pages))) + \
                          list(range(max(0, total_pages-10), total_pages))
            return ' '.join(
                reader.pages[i].extract_text()
                for i in sorted(set(target_pages))  # Remove duplicates if overlap
                if reader.pages[i].extract_text()
            )
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return ""