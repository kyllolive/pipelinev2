import os
from extract import extract_document
import glob
import argparse
import csv


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process HTML documents")
    parser.add_argument("--doc-type", required=True, help="Type of document to process")
    parser.add_argument(
        "--doc-path", required=True, help="Path to directory containing HTML files"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        html_files = glob.glob(os.path.join(args.doc_path, "*.html"))
        results = []

        for html_file in html_files:
            extraction_result = extract_document(args.doc_type, html_file)
            results.append(extraction_result)
        
        print(results)

        #Build proponent ids column    
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
