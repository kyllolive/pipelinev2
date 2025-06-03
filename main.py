import os
from extract import extract_document


def main():
    doc_type = os.getenv("DOC_TYPE")
    doc_path = os.getenv("DOC_PATH")

    try:
        extraction_result = extract_document(doc_type, doc_path)
        print(extraction_result)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
