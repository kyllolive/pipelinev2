import os

import re
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ordinance_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_ordinance_info(filepath):
    """
    Extracts ordinance numbers and year from a filepath.
    Uses parent directory year if filename doesn't contain one.
    Preserves alphabetical suffixes if they exist.
    """
    filename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    # Pre-process the filename to handle special characters and suffixes
    name_without_ext = os.path.splitext(filename)[0]

    # Handle special character suffixes first
    special_suffix_match = re.search(r"[=+]([A-Z])?$", name_without_ext)
    if special_suffix_match:
        suffix = special_suffix_match.group(1)
        name_without_ext = re.sub(r"[=+][A-Z]?$", "", name_without_ext)
        if suffix:
            name_without_ext = f"{name_without_ext}-{suffix}"

    # Remove other known suffixes
    name_without_ext = name_without_ext.replace("Pri", "")
    name_without_ext = name_without_ext.replace(".doc", "")

    # Normalize spaces and hyphens
    name_without_ext = re.sub(r"\s+", "", name_without_ext)  # Remove all spaces
    name_without_ext = re.sub(
        r"-+", "-", name_without_ext
    )  # Normalize multiple hyphens

    normalized_filename = f"{name_without_ext}.pdf"

    # Now check if the normalized filename matches our convention
    existing_format_match = re.match(
        r"^(\d{4})-(\d+)(?:-([A-Z]+))?\.pdf$", normalized_filename, re.IGNORECASE
    )
    if existing_format_match:
        return normalized_filename

    # Extract year from parent directories if they match pattern (e.g., "Ordinance 2012")
    year_from_parent = None
    parent_year_match = re.search(r"(\d{4})", parent_dir)
    if not parent_year_match:
        parent_year_match = re.search(r"(\d{4})", grandparent_dir)
    if parent_year_match:
        year_from_parent = parent_year_match.group(1)

    # Remove file extension and split by spaces and hyphens
    name_without_ext = os.path.splitext(filename)[0]

    # Replace common suffixes and problematic patterns
    name_without_ext = name_without_ext.replace(".doc", "")
    name_without_ext = name_without_ext.replace("Pri", "")  # Remove "Pri" suffix
    name_without_ext = re.sub(
        r"\s*-+\s*", "-", name_without_ext
    )  # Normalize dashes (handles -- and spaces around -)

    # Check for alphabetical suffix at the end (like -A, -B)
    suffix = None
    suffix_match = re.search(r"-([A-Z]+)$", name_without_ext)
    if suffix_match:
        suffix = suffix_match.group(1)
        # Remove the suffix for number extraction
        name_without_ext = re.sub(r"-[A-Z]+$", "", name_without_ext)

    # Extract all numbers from the filename
    numbers = re.findall(r"\d+", name_without_ext)

    # Identify year and ordinance numbers
    ordinance_numbers = []
    year = None

    for num in numbers:
        if len(num) == 4 and int(num) >= 1950:
            year = num
        # Handle numbers that might have leading zeros but are actually large numbers
        elif len(num) <= 5:  # Allow up to 5 digits for ordinance numbers
            # Remove leading zeros by converting to int and back to string
            ordinance_numbers.append(str(int(num)))

    # If no year found in filename, use parent directory year
    if not year and year_from_parent:
        year = year_from_parent

    # Log failure reasons if we can't process the file
    if not year:
        logger.warning(f"Failed to extract year from: {filepath}")
    if not ordinance_numbers:
        logger.warning(f"Failed to extract ordinance number from: {filepath}")

    # If still no year or no ordinance numbers, return None
    if not year or not ordinance_numbers:
        return None

    # Create new filename based on number of ordinance numbers (year first)
    if len(ordinance_numbers) >= 2:
        base_name = f"{year}-{ordinance_numbers[0]}-{ordinance_numbers[1]}"
    else:
        base_name = f"{year}-{ordinance_numbers[0]}"

    # Add the suffix if it exists
    if suffix:
        new_name = f"{base_name}-{suffix}.pdf"
    else:
        new_name = f"{base_name}.pdf"

    return new_name


def consolidate_pdfs(source_dir, target_dir):
    """
    Recursively finds all PDFs in source_dir and its subdirectories,
    renames them according to the specified format, and moves them to target_dir.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Count successful and failed operations
    success_count = 0
    failed_count = 0
    all_pdfs = []

    # Walk through all subdirectories and collect PDF files
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(root, filename)
                all_pdfs.append(filepath)

    logger.info(f"Found {len(all_pdfs)} PDF files in source directory")

    # Track files that have already been processed
    processed_filenames = set()
    duplicate_count = 0

    # Create failed output directory
    failed_dir = os.path.join(target_dir, "failed")
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)

    # Process each PDF file
    for filepath in all_pdfs:
        # Extract ordinance info and get new name
        original_new_name = extract_ordinance_info(filepath)
        new_name = original_new_name

        if new_name:
            # Check if this filename has already been processed
            if new_name in processed_filenames:
                duplicate_count += 1
                logger.info(
                    f"DUPLICATE DETECTED: {filepath} would create {new_name} which already exists"
                )

                # This is a duplicate - start with 'A', then 'B', ..., 'Z', then 'AA', 'AB', etc.
                base, ext = os.path.splitext(new_name)
                suffix = "A"
                suffix_name = f"{base}-{suffix}{ext}"

                while suffix_name in processed_filenames:
                    # If we reach 'Z', move to 'AA'
                    if suffix == "Z":
                        suffix = "AA"
                    # If we have a multi-letter suffix
                    elif len(suffix) > 1:
                        # Increment the last letter
                        last_char = chr(ord(suffix[-1]) + 1)
                        # If last letter goes beyond 'Z', increment the previous letter and reset last to 'A'
                        if last_char > "Z":
                            first_part = suffix[:-1]
                            if len(first_part) == 1:
                                suffix = chr(ord(first_part) + 1) + "A"
                            else:
                                # Handle more than 2 letters if needed
                                suffix = (
                                    first_part[:-1] + chr(ord(first_part[-1]) + 1) + "A"
                                )
                        else:
                            suffix = suffix[:-1] + last_char
                    else:
                        # Single letter case
                        suffix = chr(ord(suffix) + 1)

                    suffix_name = f"{base}-{suffix}{ext}"

                new_name = suffix_name

            # Add the filename (original or modified) to our processed set
            processed_filenames.add(new_name)

            # Copy the file to target directory with the appropriate name
            target_path = os.path.join(target_dir, new_name)
            shutil.copy2(filepath, target_path)

            if original_new_name != new_name:
                logger.info(f"Consolidated with suffix: {filepath} -> {target_path}")
            else:
                logger.info(f"Consolidated: {filepath} -> {target_path}")

            success_count += 1
        else:
            # Copy failed file to failed directory with original name
            failed_path = os.path.join(failed_dir, os.path.basename(filepath))
            shutil.copy2(filepath, failed_path)
            logger.error(f"Failed to parse: {filepath} -> {failed_path}")
            failed_count += 1

    logger.info(
        f"Consolidation complete. Successfully processed {success_count} files, failed to process {failed_count} files."
    )
    logger.info(f"Found {duplicate_count} duplicate files that needed suffix renaming")

    return success_count, failed_count


def count_pdfs_recursively(directory):
    """
    Recursively counts PDF files in the given directory and its subdirectories.
    """
    pdf_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_count += 1
    return pdf_count


def main():
    source_dir = (
        f"{os.getcwd()}/raw_ordinances"  # Root directory containing all ordinance subdirectories
    )
    target_dir = (
        f"{os.getcwd()}/ordinances"  # Where all PDFs will be consolidated
    )
    bucket_name = "your-gcs-bucket-name"  # Change this to your GCS bucket

    logger.info(
        f"Starting ordinance file consolidation from {source_dir} to {target_dir}"
    )

    # Consolidate all PDFs from source directory tree to target directory
    success_count, failed_count = consolidate_pdfs(source_dir, target_dir)

    logger.info("Consolidation complete.")
    logger.info(
        f"Consolidated {success_count} files, failed to process {failed_count} files."
    )
    # log files count in target directory
    pdf_count = count_pdfs_recursively(target_dir)
    logger.info(f"PDFs in target directory (recursive count): {pdf_count}")
    logger.info(f"Files in target directory (top level): {len(os.listdir(target_dir))}")

    # Create first page copies of consolidated PDFs
    # create_first_page_copies(target_dir)

    # Upload to GCS (uncomment when ready)
    # upload_to_gcs(bucket_name, target_dir)


if __name__ == "__main__":
    main()