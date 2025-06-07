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
    parser.add_argument("--output-file", required=True, help="Path to output file")

    # Parse arguments
    args = parser.parse_args()

    try:
        html_files = glob.glob(os.path.join(args.doc_path, "*.html"))
        results = []

        for html_file in html_files:
            extraction_result = extract_document(args.doc_type, html_file)
            results.append(extraction_result)
            print(extraction_result)

        with open(args.output_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
