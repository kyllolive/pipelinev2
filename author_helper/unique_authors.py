import csv
import re
import argparse
import sys
from pathlib import Path


def parse_name(full_name):
    """Parse a full name into its components."""
    name_parts = re.split(r"\s+", full_name.strip())

    # Initialize name components
    prefix = ""
    first_name = ""
    middle_initial = ""
    last_name = ""
    suffix = ""

    # Check for prefix (e.g., HON., DR.)
    while name_parts and name_parts[0].upper().endswith((".")) and len(name_parts) > 1:
        prefix = (prefix + " " + name_parts.pop(0)).strip()

    # Check for common suffixes at the end
    if name_parts and name_parts[-1].upper() in {"JR.", "SR.", "M.D.", "J.R."}:
        suffix = name_parts.pop()

    # Assign name parts
    if not name_parts:
        last_name = ""  # Handle edge case of only prefix
    elif len(name_parts) == 1:
        last_name = name_parts[0]
    else:
        first_name = name_parts[0]

        # Look for middle initial pattern (single letter followed by period)
        middle_parts = []
        last_name_parts = []

        for part in name_parts[1:]:
            if re.match(r"^[A-Z]\.$", part.upper()):
                middle_parts.append(part[0])  # Store just the letter
            else:
                last_name_parts.append(part)

        middle_initial = "".join(middle_parts)
        last_name = " ".join(last_name_parts) if last_name_parts else name_parts[-1]

    # Append suffix to last name if present
    if suffix:
        last_name = f"{last_name} {suffix}"

    return {
        "prefix": prefix,
        "firstName": first_name,
        "middleInitial": middle_initial,
        "lastName": last_name,
    }


def process_proponents(input_file, output_file, proponent_column="proponent"):
    """Process the input CSV and generate the proponents output file."""
    proponents = set()

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if proponent_column not in reader.fieldnames:
                raise ValueError(f"Column '{proponent_column}' not found in input file")

            for row in reader:
                proponent_str = row[proponent_column].strip()
                if proponent_str:
                    # Split by comma to get individual names
                    names = [
                        name.strip()
                        for name in proponent_str.split(",")
                        if name.strip()
                    ]
                    proponents.update(names)

        # Convert the set to a sorted list and process names
        output_data = []
        for cuid, full_name in enumerate(sorted(proponents), start=1):
            name_parts = parse_name(full_name)
            name_parts["cuid"] = cuid
            name_parts["fullName"] = full_name
            output_data.append(name_parts)

        # Write the output CSV
        fieldnames = [
            "cuid",
            "prefix",
            "firstName",
            "middleInitial",
            "lastName",
            "fullName",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_data)

        print(f"Successfully processed {len(output_data)} proponents")
        return True

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied accessing file", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Process proponents from a CSV file and extract name components."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="document_data.csv",
        help="Input CSV file path (default: document_data.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="new_proponents.csv",
        help="Output CSV file path (default: proponents.csv)",
    )
    parser.add_argument(
        "-c",
        "--column",
        default="proponent",
        help="Name of the proponent column in the input CSV (default: proponent)",
    )

    args = parser.parse_args()

    # Convert to Path objects for better path handling
    input_path = Path(args.input)
    output_path = Path(args.output)

    success = process_proponents(input_path, output_path, args.column)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
