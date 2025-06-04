1. Run script to rename filename of pdfs
2. Run olmocr to get html for each documents
3. Generate thumbnails
4. RUN `python main.py --doc-type "ordinance" --doc-path "./ordinances_html" --output-file "ordinances-1.csv"`
5. Get all unique proponents, normalize it (for now manually assign repeating or similar names the same proponent ids)
6. Classify
7. Summarize
8 done.