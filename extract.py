from bs4 import BeautifulSoup
import os
from utils import (
    extract_document_type,
    extract_ordinance_number,
    extract_resolution_number,
    extract_number_from_filename,
    extract_title,
    extract_proponent,
    extract_date,
    extract_detected_text,
)


def extract_numbers_from_text(p_text, text_content, result):
    """Extract numbers if they haven't been found yet."""
    if result["ordinance_number"] is None:
        result["ordinance_number"] = extract_ordinance_number(p_text, text_content)
    if result["resolution_number"] is None:
        result["resolution_number"] = extract_resolution_number(p_text, text_content)
    return (
        result["ordinance_number"] is not None
        and result["resolution_number"] is not None
    )


def extract_document(document_type, document_path):
    with open(document_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "lxml")
    text_divs = soup.select(".text")
    text_content = ""
    for div in text_divs:
        text_content += div.get_text()

    filename = os.path.basename(document_path).replace(".html", "")
    # Remove prefix and suffix to get just the year-number
    filename = filename.replace("olmocr-codification_ordinances_", "")
    filename = filename.replace("olmocr-codification_resolutions_", "")
    filename = filename.replace("_pdf", "")
    result = {
        "filename": filename,
        "detected_text": None,
        "ordinance_number": None,
        "resolution_number": None,
        "document_type": None,
        "title": None,
        "proponent": None,
        "date_enacted": None,
        "date_note": None,
    }

    result["document_type"] = extract_document_type(document_type).lower()
    def_res = extract_number_from_filename(document_path)

    if document_type == "other":
        result.update(def_res)
        return result

    for p in soup.find_all("p"):
        if extract_numbers_from_text(p.get_text().strip(), text_content, result):
            break

    if result["ordinance_number"] is None and result["resolution_number"] is None:
        result.update(def_res)

    if result["title"] is None:
        extract_title(soup, text_content, result)

    if result["proponent"] is None:
        extract_proponent(soup, text_content, result)

    if result["date_enacted"] is None:
        extract_date(text_content, result, document_path)

    if result["detected_text"] is None:
        extract_detected_text(soup, result)

    return result
