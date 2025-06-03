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


def extract_title(soup, text_content, result):
    """
    Extract the title based on document type with specific rules.

    Args:
        p_text: Current paragraph text
        text_content: Full document text content
        result: Result dictionary containing numbers and other metadata
        document_type: Type of document ('ordinance' or 'resolution')
        filename: Original filename for fallback
    """
    title_prefixes = ["AN ORDINANCE", "A RESOLUTION"]

    title = None

    for p in soup.find_all("p"):
        p_text = p.get_text().strip()
        for prefix in title_prefixes:
            if prefix in p_text.upper():
                title = p_text
                break
        if title:
            result["title"] = title
            return True
        
    if title is None:
        title_match = None
        for prefix in title_prefixes:
            pattern = rf"{prefix}(.*?)(?:Be it|SECTION 1|\n\n|WHEREAS)"
            match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
            if match:
                title_match = match
                break
        if title_match:
            title_text = title_match.group(1).strip()
            title = f"{prefix} {title_text}"
            clean_title = re.sub(r"\s+", " ", title)
            result["title"] = clean_title
            return True

    # Fallback: Try to construct title from document number if available
    if result.get("document_type") == "ordinance":
        result["title"] = f"ORDINANCE NO. {result['ordinance_number']}"
        return True
    elif result.get("document_type") == "resolution":
        result["title"] = f"RESOLUTION NO. {result['resolution_number']}"
        return True

    return False

def extract_proponent(soup, text_content, result):
    """
    Extract the proponent from the document text.
    """
    # First try: Direct keyword matches
    proponent = None

    proponent_keywords = [
        "Author:",
        "AUTHORS:",
        "Authors:",
        "AUTHOR:",
        "Proponent:",
        "Proponents:",
        "PROPONENTS:",
        "PROPONENT:",
    ]
    for p in soup.find_all("p"):
        p_text = p.get_text().strip()

  
        if any(keyword in p_text for keyword in proponent_keywords):
    
            for keyword in proponent_keywords:
                if keyword in p_text:
                    p_text = p_text.replace(keyword, "").strip()
                    break
            hon_pattern = r"(HON\.\s+[A-Z][A-Za-z\.\s,]+?)(?=,?\s*HON\.|$)"
            hon_matches = re.findall(hon_pattern, p_text)

            if hon_matches:
         
                result["proponent"] = ", ".join(hon_matches)
            else:
                result["proponent"] = None 
            break
    if result["proponent"] is None:
        for p in soup.find_all("p"):
            p_text = p.get_text().strip()
            if (
                "after due deliberation" in p_text.lower()
                and "on motion of" in p_text.lower()
            ):
                print(f"Found target paragraph: {p_text[:100]}...")

                # Get the text after "on motion of"
                start_phrase = "on motion of"
                start_pos = p_text.lower().find(start_phrase) + len(start_phrase)

                # Look for various phrases that would end the proponent's name
                end_phrases = [
                    "seconded by",
                    "duly seconded",
                    "duly",
                    "duty",
                    ", seconded",
                    ", and seconded",
                    ", resolved",
                    "and adopted",
                    ", be it resolved",
                ]

                end_pos = len(p_text) 

             
                for phrase in end_phrases:
                    pos = p_text.lower().find(phrase, start_pos)
                    if pos > 0 and pos < end_pos:
                        end_pos = pos

                if start_pos > 0 and end_pos > start_pos:
              
                    proponent_text = p_text[start_pos:end_pos].strip()

                    proponent_text = proponent_text.strip(",").strip()

                    # Ensure we have proper naming convention (Hon.)
                    if " and hon." in proponent_text.lower():
                        
                        proponents = []
                        for part in proponent_text.split(" and "):
                            
                            if "hon" in part.lower():
                                proponents.append(part.strip())

                        if proponents:
                            result["proponent"] = ", ".join(proponents).upper()
                            print(
                                f"Extracted multiple proponents: {result['proponent']}"
                            )
                            break
                    elif "hon" in proponent_text.lower():
                        
                        proponent_text = " ".join(proponent_text.split())
                        result["proponent"] = proponent_text.strip().upper()
                        print(f"Extracted proponent: {result['proponent']}")
                        break

    if result["proponent"] is None:
        result["proponent"] = None


def extract_date(text_content, result, filename):
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
    
    year_match = re.search(r"(\d{4})", filename)
    year = year_match.group(1) if year_match else None

    if not result["date_enacted"] and year:
        result["date_enacted"] = f"{year}-01-01"
        result["date_note"] = "Estimated based on year only."
        return True
    return False

def extract_detected_text(soup, result):
    """
    Extract the detected text from the document.
    """
    word_limit = 1500
    document_div = soup.find("div", class_="document")
    if document_div:
    
        p_tags = document_div.find_all("p")

        all_paragraphs = [p.get_text(strip=True) for p in p_tags]

    
        full_text = "\n\n".join(all_paragraphs)
        words = full_text.split()

        if len(words) > word_limit:
            limited_text = " ".join(words[:word_limit])
            
            result["detected_text"] = limited_text
            return True
        else:
            result["detected_text"] = full_text
            return True

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
