from bs4 import BeautifulSoup
import os
from utils import extract_document_type


def extract_document(document_type, document_path):
    with open(document_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "lxml")

    text_divs = soup.select(".text")
    text_content = ""
    for div in text_divs:
        text_content += div.get_text()

    # Extraction

    document_type = extract_document_type(document_type)
