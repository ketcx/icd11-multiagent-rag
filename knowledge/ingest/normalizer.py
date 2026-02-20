"""Functions for cleaning and normalizing extracted text."""

import re

def normalize_text(text: str) -> str:
    """Normalizes text extracted from the ICD-11 PDF.
    
    1. Removes line-break hyphens
    2. Normalizes multiple spaces
    """
    # Remove end of line hyphens that split words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text
