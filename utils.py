import re
from datetime import datetime
import os


def extract_number_from_filename(filename):
    """
    Extract the resolution or ordinance number from the filename as a fallback.
    """
    result = {
        "ordinance_number": None,
        "resolution_number": None,
    }
    file_path = os.path.basename(filename)

    # Handle 2009 files with errors specially
    if re.search(r"res-dataset_2009-\d+-[_¬åL]", file_path):
        pass
    else:
        # Case 1: Resolution file pattern (res-dataset_YYYY_NN)
        res_match = re.search(r"res-dataset_(\d{4}-\d+)_", file_path)

        if res_match:
            result["resolution_number"] = f"{res_match.group(1)}"
        # Case 2: Ordinance file pattern (dataset_YYYY_NN)
        ord_match = re.search(r"dataset_(\d{4}-\d+)_", file_path)
        if ord_match:
            result["ordinance_number"] = f"{ord_match.group(1)}"

    return result


def extract_document_type(type):
    """
    Extract the document type from the document path.
    """
    return type


def extract_ordinance_number(p_text, text_content):
    """
    Extract the ordinance number from the document path.
    """

    if "ORDINANCE NO." in p_text.upper():
        # Extract the number after "ORDINANCE NO."
        ord_match = re.search(r"ORDINANCE NO\.?\s+([\d-]+)", p_text, re.IGNORECASE)
        if ord_match:
            return ord_match.group(1)

    ord_match = re.search(r"ORDINANCE NO\.?\s+([\d-]+)", text_content, re.IGNORECASE)
    if ord_match:
        return ord_match.group(1)

    return None


def extract_resolution_number(p_text, text_content):
    """
    Extract the resolution number from the document path.
    """
    if "RESOLUTION NO." in p_text.upper():
        p_text = p_text.strip()

        res_match = re.search(r"RESOLUTION NO\.?\s+([\d-]+)", p_text, re.IGNORECASE)
        if res_match:
            return res_match.group(1)

    res_match = re.search(r"RESOLUTION NO\.?\s+([\d-]+)", text_content, re.IGNORECASE)
    if res_match:
        return res_match.group(1)

    return None


def extract_title(p_text, text_content, result, document_type, filename):
    """
    Extract the title based on document type with specific rules.

    Args:
        p_text: Current paragraph text
        text_content: Full document text content
        result: Result dictionary containing numbers and other metadata
        document_type: Type of document ('ordinance' or 'resolution')
        filename: Original filename for fallback
    """

    def clean_title(title):
        """Clean and standardize title text."""
        # Remove multiple whitespace and normalize
        title = re.sub(r"\s+", " ", title).strip()
        # Remove common noise patterns
        title = re.sub(r"[\r\n]+", " ", title)
        # Remove duplicate prefixes
        title = re.sub(
            r"(AN ORDINANCE|A RESOLUTION)\s+\1", r"\1", title, flags=re.IGNORECASE
        )
        return title

    def extract_ordinance_title():
        """Extract title for ordinance documents."""
        # Look for the full ordinance title pattern
        patterns = [
            r"AN ORDINANCE\s+(.*?)(?=(?:BE IT ORDAINED|SECTION 1\.|\n\n|WHEREAS|;))",
            r"AN ORDINANCE\s+(.*?)(?=\n\n)",
            r"AN ORDINANCE\s+(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
            if match:
                title = f"AN ORDINANCE {match.group(1).strip()}"
                return clean_title(title)

        if result["ordinance_number"]:
            return f"AN ORDINANCE {result['ordinance_number']}"

        # Last resort: use filename
        return f"AN ORDINANCE {os.path.splitext(os.path.basename(filename))[0]}"

    def extract_resolution_title():
        patterns = [
            r"A RESOLUTION\s+(.*?)(?=(?:BE IT RESOLVED|SECTION 1\.|\n\n|WHEREAS|;))",
            r"A RESOLUTION\s+(.*?)(?=\n\n)",
            r"A RESOLUTION\s+(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
            if match:
                title = f"A RESOLUTION {match.group(1).strip()}"
                return clean_title(title)

        if result["resolution_number"]:
            return f"RESOLUTION NO. {result['resolution_number']}"

        return None

    document_type = document_type.lower()

    if document_type == "ordinance":
        result["title"] = extract_ordinance_title()
        return True
    elif document_type == "resolution":
        title = extract_resolution_title()
        if title:
            result["title"] = title
            return True

    return False


def extract_proponent(p_text, text_content, result):
    """
    Extract the proponent from the document text.
    """

    def debug_print(msg, text_sample):
        print(f"DEBUG - {msg}: {text_sample[:200]}...")

    debug_print("Processing paragraph", p_text)

    # First try: Direct keyword matches
    proponent_keywords = [
        "Author:",
        "AUTHORS:",
        "Authors:",
        "AUTHOR:",
        "Proponent:",
        "Proponents:",
        "PROPONENTS:",
        "PROPONENT:",
        "By Hon.",  # Adding common pattern
        "BY HON.",
    ]

    # Check for keyword matches in both p_text and full text_content
    for keyword in proponent_keywords:
        # Check in current paragraph
        if keyword in p_text:
            debug_print(f"Found keyword {keyword} in paragraph", p_text)
            text_after_keyword = p_text[p_text.find(keyword) + len(keyword) :].strip()
            hon_pattern = r"(?:HON\.|Hon\.|hon\.|BY HON\.|By Hon\.)\s*([A-Z][A-Za-z\.\s,]+?)(?=(?:,?\s*(?:HON\.|Hon\.|and|AND|$)))"
            hon_matches = re.findall(hon_pattern, text_after_keyword, re.IGNORECASE)

            if hon_matches:
                result["proponent"] = ", ".join(
                    name.strip().upper() for name in hon_matches
                )
                debug_print("Extracted proponents from keyword", result["proponent"])
                return True

        # Check in full text if not found in paragraph
        if keyword in text_content:
            debug_print(f"Found keyword {keyword} in full text", text_content)
            lines = text_content.split("\n")
            for line in lines:
                if keyword in line:
                    hon_pattern = r"(?:HON\.|Hon\.|hon\.|BY HON\.|By Hon\.)\s*([A-Z][A-Za-z\.\s,]+?)(?=(?:,?\s*(?:HON\.|Hon\.|and|AND|$)))"
                    hon_matches = re.findall(hon_pattern, line, re.IGNORECASE)
                    if hon_matches:
                        result["proponent"] = ", ".join(
                            name.strip().upper() for name in hon_matches
                        )
                        debug_print(
                            "Extracted proponents from full text", result["proponent"]
                        )
                        return True

    # Second try: Motion pattern
    motion_patterns = [
        r"(?:on\s+motion\s+of|moved\s+by)\s+((?:HON\.|Hon\.|hon\.)\s*[A-Z][A-Za-z\.\s,]+?)(?=(?:,|\sand\s|seconded|duly|resolved))",
        r"(?:after\s+due\s+deliberation\s+(?:and\s+)?on\s+motion\s+of)\s+((?:HON\.|Hon\.|hon\.)\s*[A-Z][A-Za-z\.\s,]+?)(?=(?:,|\sand\s|seconded|duly|resolved))",
    ]

    for pattern in motion_patterns:
        # Try in current paragraph
        matches = re.findall(pattern, p_text, re.IGNORECASE)
        if matches:
            debug_print("Found motion pattern match in paragraph", matches[0])
            result["proponent"] = ", ".join(name.strip().upper() for name in matches)
            return True

        # Try in full text
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        if matches:
            debug_print("Found motion pattern match in full text", matches[0])
            result["proponent"] = ", ".join(name.strip().upper() for name in matches)
            return True

    debug_print("No proponent found in", p_text)
    return False


def extract_date(text_content, result):
    """
    Extract the date from the document path.
    """
    date_patterns = [
        r"(?:APPROVED:?|ENACTED:?|PASSED:?).*?(\w+\s+\d{1,2},\s*\d{4})",  # June 10, 2020
        r"(?:SESSION HALL ON|APPROVED\.|ADOPTED:?|ENACTED ON).*?(\w+\s+\d{1,2},\s*\d{4})",  # Session hall on June 10, 2020
        r"(?:SESSION HALL ON|APPROVED\.|ADOPTED:?|ENACTED ON).*?(\d{1,2}/\d{1,2}/\d{2,4})",  # Session hall on 7/10/2020
        r"(\d{1,2}/\d{1,2}/\d{2,4})",  # 7/10/60 or 7/10/1960
        r"(\d{1,2}-\d{1,2}-\d{2,4})",  # 7-10-60 or 7-10-1960
        r"dated\s+(\w+\s+\d{1,2},\s*\d{4})",  # dated June 10, 2020
        r"this\s+(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+\w+,\s*\d{4})",  # this 10th day of June, 2020
        r"ENACTED,\s*\n*\s*(\w+\s+\d{1,2},\s*\d{4})",  # ENACTED, July 10, 2024
        r"ENACTED,\s*\n*\s*(\d{1,2}/\d{1,2}/\d{2,4})",  # ENACTED, 7/10/2024
    ]
    for pattern in date_patterns[:1]:
        date_match = re.search(pattern, text_content, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            try:
                # Try to parse and standardize the date
                date_obj = parse_date(date_str)
                if date_obj:
                    result["date_enacted"] = date_obj.strftime("%Y-%m-%d")
                    return True
            except ValueError:
                result["date_enacted"] = date_str
                return False
    if not result["date_enacted"]:
        for pattern in date_patterns[1:]:
            date_matches = re.findall(pattern, text_content, re.IGNORECASE)
            if date_matches:
                for date_str in date_matches:
                    try:
                        date_obj = parse_date(date_str)
                        if date_obj:
                            result["date_enacted"] = date_obj.strftime("%Y-%m-%d")
                            return True
                    except ValueError:
                        continue
    return False


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
