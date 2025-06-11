#!/usr/bin/env python3
"""
CSV Merger Script - Add Summary Column
Merges two CSV files by adding the 'summary' column from the second CSV to the first CSV.
Matches rows based on the 'filename' column.

Usage:
    python csv_merger.py <primary_csv> <secondary_csv> [output_csv]

Where:
    primary_csv: CSV file with classifications (missing summary column)
    secondary_csv: CSV file with summary column
    output_csv: Output filename (optional, defaults to 'merged_output.csv')
"""

import pandas as pd
import sys
import os
from pathlib import Path


def validate_file(filepath, description):
    """Validate that a file exists and is readable."""
    if not os.path.exists(filepath):
        print(f"Error: {description} file '{filepath}' does not exist.")
        return False

    if not os.path.isfile(filepath):
        print(f"Error: '{filepath}' is not a file.")
        return False

    try:
        # Try to read a few lines to validate it's a proper CSV
        pd.read_csv(filepath, nrows=1)
        return True
    except Exception as e:
        print(f"Error: Cannot read {description} file '{filepath}': {e}")
        return False


def load_csv_safely(filepath, description):
    """Load CSV file with error handling."""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {description}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading {description} from '{filepath}': {e}")
        return None


def merge_csvs(primary_csv, secondary_csv, output_csv="merged_output.csv"):
    """
    Merge two CSV files by adding summary column from secondary to primary.

    Args:
        primary_csv (str): Path to primary CSV (with classifications)
        secondary_csv (str): Path to secondary CSV (with summary)
        output_csv (str): Path for output CSV file

    Returns:
        bool: True if successful, False otherwise
    """

    # Validate input files
    print("Validating input files...")
    if not validate_file(primary_csv, "Primary CSV"):
        return False
    if not validate_file(secondary_csv, "Secondary CSV"):
        return False

    # Load CSV files
    print("\nLoading CSV files...")
    primary_df = load_csv_safely(primary_csv, "primary CSV")
    if primary_df is None:
        return False

    secondary_df = load_csv_safely(secondary_csv, "secondary CSV")
    if secondary_df is None:
        return False

    # Check for required columns
    print("\nValidating columns...")
    if "filename" not in primary_df.columns:
        print("Error: Primary CSV is missing 'filename' column")
        return False

    if "filename" not in secondary_df.columns:
        print("Error: Secondary CSV is missing 'filename' column")
        return False

    if "summary" not in secondary_df.columns:
        print("Error: Secondary CSV is missing 'summary' column")
        return False

    print("✓ All required columns found")

    # Display column information
    print(f"\nPrimary CSV columns: {list(primary_df.columns)}")
    print(f"Secondary CSV columns: {list(secondary_df.columns)}")

    # Check if summary already exists in primary
    if "summary" in primary_df.columns:
        print(
            "\nWarning: Primary CSV already has a 'summary' column. It will be overwritten."
        )

    # Clean filename columns (remove whitespace)
    primary_df["filename"] = primary_df["filename"].astype(str).str.strip()
    secondary_df["filename"] = secondary_df["filename"].astype(str).str.strip()

    # Create summary lookup dictionary
    summary_dict = dict(zip(secondary_df["filename"], secondary_df["summary"]))
    print(f"\nCreated summary lookup for {len(summary_dict)} unique filenames")

    # Add summary column to primary dataframe
    primary_df["summary"] = primary_df["filename"].map(summary_dict)

    # Count matches
    matched_count = primary_df["summary"].notna().sum()
    total_count = len(primary_df)

    print(f"\nMerge Results:")
    print(f"  Total rows: {total_count}")
    print(f"  Rows with summary: {matched_count}")
    print(f"  Rows without summary: {total_count - matched_count}")
    print(f"  Match rate: {matched_count / total_count * 100:.1f}%")

    # Show unmatched filenames if any
    if matched_count < total_count:
        unmatched = primary_df[primary_df["summary"].isna()]["filename"].unique()
        print(f"\nUnmatched filenames ({len(unmatched)}):")
        for filename in unmatched[:10]:  # Show first 10
            print(f"  - '{filename}'")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # Fill NaN summaries with empty string
    primary_df["summary"] = primary_df["summary"].fillna("")

    # Save merged data
    try:
        primary_df.to_csv(output_csv, index=False)
        print(f"\n✓ Merged CSV saved to: {output_csv}")
        print(f"  File size: {os.path.getsize(output_csv):,} bytes")

        # Show preview of merged data
        print(f"\nPreview of merged data (first 3 rows):")
        preview_cols = ["filename", "document_type", "title", "date_enacted"]
        if "summary" in primary_df.columns:
            preview_cols.append("summary")

        # Only show columns that exist
        available_cols = [col for col in preview_cols if col in primary_df.columns]

        for i, row in primary_df.head(3).iterrows():
            print(f"\nRow {i + 1}:")
            for col in available_cols:
                value = str(row[col])
                if col == "summary" and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {col}: {value}")

        return True

    except Exception as e:
        print(f"Error saving merged CSV: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute merge."""

    print("CSV Merger - Add Summary Column")
    print("=" * 40)

    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python csv_merger.py <primary_csv> <secondary_csv> [output_csv]")
        print("\nWhere:")
        print("  primary_csv: CSV file with classifications (missing summary column)")
        print("  secondary_csv: CSV file with summary column")
        print(
            "  output_csv: Output filename (optional, defaults to 'merged_output.csv')"
        )
        print("\nExample:")
        print("  python csv_merger.py data1.csv data2.csv merged_data.csv")
        sys.exit(1)

    primary_csv = sys.argv[1]
    secondary_csv = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "merged_output.csv"

    print(f"Primary CSV: {primary_csv}")
    print(f"Secondary CSV: {secondary_csv}")
    print(f"Output CSV: {output_csv}")
    print()

    # Check if output file already exists
    if os.path.exists(output_csv):
        response = input(
            f"Output file '{output_csv}' already exists. Overwrite? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            sys.exit(0)

    # Perform the merge
    success = merge_csvs(primary_csv, secondary_csv, output_csv)

    if success:
        print(f"\n🎉 Success! Merged CSV created at: {output_csv}")
        sys.exit(0)
    else:
        print("\n❌ Merge failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
