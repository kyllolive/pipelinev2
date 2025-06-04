import csv
import re
import os


def normalize_name(name):
    """Normalize a name by removing common suffixes and standardizing format."""
    # Remove common suffixes
    suffixes = [" JR.", " SR.", " M.D.", " J.R."]
    name = name.upper().strip()
    for suffix in suffixes:
        name = name.replace(suffix.upper(), "")
    return name.strip()


# Get correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
unique_authors_path = os.path.join(script_dir, "unique_authors.csv")
ordinances_input_path = os.path.join(parent_dir, "ordinances-1.csv")
ordinances_output_path = os.path.join(parent_dir, "ordinances-1-cuid.csv")

# First, load the name mappings from unique_authors.csv into a dictionary
name_to_id = {}
with open(unique_authors_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Create a normalized version of the full name for comparison
        normalized_name = normalize_name(row["fullName"])
        name_to_id[normalized_name] = row["cuid"]
        print(f"Loaded mapping: {normalized_name} -> {row['cuid']}")  # Debug print


def process_proponents(proponents_str):
    if not proponents_str:
        return None

    # Split multiple proponents by comma
    proponents = [p.strip() for p in proponents_str.split(",")]
    print(f"Processing proponents: {proponents}")  # Debug print

    # Find matching IDs
    ids = []
    for proponent in proponents:
        # Normalize the proponent name
        normalized_proponent = normalize_name(proponent)
        print(f"Looking for match for: {normalized_proponent}")  # Debug print

        # Look for an exact match first
        found = False
        for name, id in name_to_id.items():
            # Try exact match first
            if normalized_proponent == name:
                ids.append(int(id))
                found = True
                print(
                    f"Found exact match: {normalized_proponent} -> {id}"
                )  # Debug print
                break

        # If no exact match, try partial match but be more strict
        if not found:
            for name, id in name_to_id.items():
                name_parts = set(name.split())
                proponent_parts = set(normalized_proponent.split())
                # Check if the core name parts match (excluding prefix/suffix)
                if len(name_parts & proponent_parts) >= 3:  # At least 3 parts match
                    ids.append(int(id))
                    print(
                        f"Found partial match: {normalized_proponent} -> {id}"
                    )  # Debug print
                    break
            if not found:
                print(f"No match found for: {normalized_proponent}")  # Debug print

    return f"[{','.join(str(id) for id in ids)}]" if ids else None


# Process the main CSV file
with (
    open(ordinances_input_path, "r") as infile,
    open(ordinances_output_path, "w", newline="") as outfile,
):
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read header row
    header = next(reader)
    print(f"Input CSV headers: {header}")  # Debug print

    # Find the proponent column index
    proponent_column_index = header.index(
        "proponent"
    )  # or whatever the actual column name is

    # Write header row with new column
    writer.writerow(header + ["proponent_ids"])

    # Process each row
    for row in reader:
        if len(row) > 0:  # Make sure row has content
            proponent_str = (
                row[proponent_column_index] if row[proponent_column_index] else ""
            )
            processed_proponents = process_proponents(proponent_str)
            print(
                f"Processing row - Proponent: {proponent_str} -> IDs: {processed_proponents}"
            )  # Debug print
            # Add the new column while keeping the original
            writer.writerow(
                row + [processed_proponents if processed_proponents else ""]
            )
