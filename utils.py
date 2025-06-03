import re
from datetime import datetime


def extract_clean_text(text):
    """
    Clean the text by removing extra whitespace and newlines.
    """
    return text.strip()


def extract_document_type(type):
    """
    Extract the document type from the document path.
    """
    return type


def extract_ordinance_number(document_path):
    """
    Extract the ordinance number from the document path.
    """
    return document_path.split("/")[-1].split(".")[0]


def extract_resolution_number(document_path):
    """
    Extract the resolution number from the document path.
    """
    return document_path.split("/")[-1].split(".")[0]


def extract_title(document_path):
    """
    Extract the title from the document path.
    """
    return document_path.split("/")[-1].split(".")[0]


def extract_proponent(document_path):
    """
    Extract the proponent from the document path.
    """
    return document_path.split("/")[-1].split(".")[0]


def extract_date(document_path):
    """
    Extract the date from the document path.
    """
    return document_path.split("/")[-1].split(".")[0]


def parse_date(date_str):
    """
    Parse a date string in various formats and return a datetime object.

    Args:
        date_str: Date string to parse

    Returns:
        datetime: Parsed date object or None if parsing fails
    """
    formats = [
        "%m/%d/%y",  # 7/10/60
        "%m/%d/%Y",  # 7/10/1960
        "%B %d, %Y",  # June 10, 2020
        "%b %d, %Y",  # Jun 10, 2020
        "%d-%m-%Y",  # 10-06-2020
        "%d-%m-%y",  # 10-06-20
        "%B %d %Y",  # June 10 2020
        "%d %B, %Y",  # 10 June, 2020
        "%d %B %Y",  # 10 June 2020
        "%dth day of %B, %Y",  # 10th day of June, 2020
    ]

    # Remove ordinal indicators
    date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def standardize_document_number(number, doc_type, year=None):
    """
    Standardize ordinance/resolution numbers to a sortable format.

    Args:
        number: The original number string
        doc_type: 'ordinance' or 'resolution'
        year: Optional year to use if not present in the number

    Returns:
        str: Standardized number in format NN-YYYY or NN-NNN-YYYY format
    """
    if not number:
        return None

    # Extract year if present in the number
    year_in_number = None

    # Check for YYYY-NN or YYYY-NN-NNN format
    year_prefix_match = re.match(r"^(\d{4})-(.+)$", number)
    if year_prefix_match:
        year_in_number = year_prefix_match.group(1)
        num = year_prefix_match.group(2)
        return f"{num}-{year_in_number}"

    # Check for NN-YYYY format
    year_suffix_match = re.match(r"^(.+)-(\d{4})$", number)
    if year_suffix_match:
        num = year_suffix_match.group(1)
        year_in_number = year_suffix_match.group(2)
        return f"{num}-{year_in_number}"  # Already in desired format

    # If we have a simple number without year (50-A, 243, 120)
    # and we have a year from context (filename or content)
    if year:
        # Try to extract year from filename if it's in the format YYYY
        if re.match(r"^\d{4}$", str(year)):
            return f"{number}-{year}"

    # If we can't standardize, return as is
    return number
