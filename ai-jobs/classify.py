import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import gc
import os
import re
import time


def classify_batch_optimized(
    model, tokenizer, titles, categories, subcategories, document_type, device
):
    """
    Optimized batch classification with single model call per title.
    Returns results and performance statistics.
    """
    # Determine default subcategory based on document type
    default_subcategory = (
        "GENERAL_ORDINANCE"
        if document_type.lower() == "ordinance"
        else "GENERAL_RESOLUTION"
    )

    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    batch_start_time = time.time()

    for title in titles:
        # Single comprehensive prompt that does everything at once
        prompt = f"""You are an expert municipal ordinance classifier. Analyze and classify this ordinance title:

TITLE: "{title}"

AVAILABLE CATEGORIES: {", ".join(categories[:-1])}

AVAILABLE DOCUMENT TYPES: {", ".join(subcategories)}

TASK: Provide classification in this EXACT format:
CATEGORIES: [list 1-3 most relevant categories, comma-separated]
TYPE: [single most appropriate document type]
CONFIDENCE: [0-100 score for overall classification confidence]

RULES:
- Select 1-3 categories that best represent the ordinance's purpose
- Choose ONE document type that best fits
- If no categories fit, use "NO_CLASSIFICATION"
- If no document type fits specifically, use "{default_subcategory}"
- Be concise and direct

Example response:
CATEGORIES: ENVIRONMENT, PUBLIC_HEALTH_AND_SANITATION
TYPE: GENERAL_ORDINANCE
CONFIDENCE: 85"""

        # Single model call
        input_data = {"role": "user", "content": prompt}
        input_ids = tokenizer.apply_chat_template(
            [input_data],
            return_tensors="pt",
            thinking=False,
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)

        # Track input tokens
        input_token_count = input_ids["input_ids"].shape[1]
        total_input_tokens += input_token_count

        set_seed(42)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=150,
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.eos_token_id,
            )

        # Track output tokens
        output_token_count = output.shape[1] - input_token_count
        total_output_tokens += output_token_count

        response = tokenizer.decode(
            output[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Parse response
        categories_match = re.search(r"CATEGORIES:\s*(.+)", response, re.IGNORECASE)
        type_match = re.search(r"TYPE:\s*(.+)", response, re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response, re.IGNORECASE)

        # Extract categories
        if categories_match:
            category_text = categories_match.group(1).strip()
            extracted_categories = [
                cat.strip() for cat in category_text.split(",") if cat.strip()
            ]
            # Validate categories exist in our list
            valid_categories = [
                cat
                for cat in extracted_categories
                if cat in categories or cat == "NO_CLASSIFICATION"
            ]
            if not valid_categories:
                valid_categories = ["NO_CLASSIFICATION"]
        else:
            valid_categories = ["NO_CLASSIFICATION"]

        # Extract document type
        if type_match:
            doc_type = type_match.group(1).strip()
            if doc_type not in subcategories:
                doc_type = default_subcategory
        else:
            doc_type = default_subcategory

        # Extract confidence
        confidence = int(confidence_match.group(1)) if confidence_match else 75

        all_results.append(
            {
                "categories": valid_categories,
                "subcategory": doc_type,
                "confidence": confidence,
                "raw_response": response,
            }
        )

    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    total_tokens = total_input_tokens + total_output_tokens

    # Performance statistics
    stats = {
        "batch_duration": batch_duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / batch_duration if batch_duration > 0 else 0,
        "input_tokens_per_second": total_input_tokens / batch_duration
        if batch_duration > 0
        else 0,
        "output_tokens_per_second": total_output_tokens / batch_duration
        if batch_duration > 0
        else 0,
        "titles_processed": len(titles),
        "avg_tokens_per_title": total_tokens / len(titles) if titles else 0,
    }

    return all_results, stats


def classify_ordinances_batch(
    model,
    tokenizer,
    titles,
    categories,
    subcategories,
    document_type,
    device,
    batch_size=16,
):
    """Optimized batch processing with larger batch sizes."""
    all_classifications = []
    all_subcategories = []
    all_confidence_scores = []
    all_diagnostic_notes = []

    # Process in larger batches
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(titles) + batch_size - 1) // batch_size}: {len(batch_titles)} titles"
        )

        try:
            # Process batch
            batch_results, batch_stats = classify_batch_optimized(
                model,
                tokenizer,
                batch_titles,
                categories,
                subcategories,
                document_type,
                device,
            )

            # Log performance statistics
            print(f"  📊 Batch Performance:")
            print(f"     • Duration: {batch_stats['batch_duration']:.2f}s")
            print(f"     • Total tokens: {batch_stats['total_tokens']:,}")
            print(f"     • Input tokens: {batch_stats['total_input_tokens']:,}")
            print(f"     • Output tokens: {batch_stats['total_output_tokens']:,}")
            print(f"     • Tokens/sec: {batch_stats['tokens_per_second']:.1f}")
            print(
                f"     • Input tokens/sec: {batch_stats['input_tokens_per_second']:.1f}"
            )
            print(
                f"     • Output tokens/sec: {batch_stats['output_tokens_per_second']:.1f}"
            )
            print(f"     • Avg tokens/title: {batch_stats['avg_tokens_per_title']:.1f}")

            # Process results
            for result in batch_results:
                categories_str = ", ".join(result["categories"])
                subcategory_str = result["subcategory"]
                confidence = result["confidence"]

                # Create diagnostic note
                diagnostic_note = f"Categories: {categories_str} | Type: {subcategory_str} | Confidence: {confidence}"

                # Store results
                all_classifications.append(categories_str)
                all_subcategories.append(subcategory_str)
                all_confidence_scores.append(confidence)
                all_diagnostic_notes.append(diagnostic_note)

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            # Add fallback values for the entire batch
            default_subcategory = (
                "GENERAL_ORDINANCE"
                if document_type.lower() == "ordinance"
                else "GENERAL_RESOLUTION"
            )
            for _ in batch_titles:
                all_classifications.append("NO_CLASSIFICATION")
                all_subcategories.append(default_subcategory)
                all_confidence_scores.append(0)
                all_diagnostic_notes.append(f"Error: {str(e)}")

        # Less frequent memory cleanup
        if i % (batch_size * 5) == 0:  # Every 5 batches instead of every 10 items
            torch.cuda.empty_cache()
            gc.collect()

    return (
        all_classifications,
        all_subcategories,
        all_confidence_scores,
        all_diagnostic_notes,
    )


def main():
    # Set up device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load the model with optimizations for 48GB VRAM
    model_path = "ibm-granite/granite-3.3-2b-instruct"
    print("Loading model and tokenizer...")

    # Configure for high performance with speed optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="cuda",  # Direct GPU mapping
        use_cache=True,  # Use KV cache for faster generation
        # Quantization for speed (uncomment if you want to trade some accuracy for speed)
        # load_in_8bit=True,  # 8-bit quantization for 2x speed boost
        # load_in_4bit=True,  # 4-bit quantization for 4x speed boost (requires bitsandbytes)
        trust_remote_code=True,  # Allow custom model code
    )

    # Move model to eval mode
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define the categories
    categories = [
        "AGRICULTURE_AND_FISHERIES",
        "ANIMAL_WELFARE",
        "CHILD_AND_YOUTH",
        "CONSUMER_PROTECTION_AND_COMMERCE",
        "CULTURE_AND_HERITAGE",
        "DIGITAL_GOVERNANCE_AND_SERVICES",
        "DISASTER_RISK_REDUCTION",
        "EDUCATION",
        "ENVIRONMENT",
        "GAMES_AND_RECREATION",
        "GENDER_AND_DEVELOPMENT",
        "HOUSING",
        "INDIGENOUS_AND_MINORITY_GROUPS",
        "INFRASTRUCTURE_AND_PUBLIC_WORKS",
        "INNOVATION_AND_TECHNOLOGY",
        "LAND_USE_AND_ZONING",
        "LIVELIHOOD_AND_EMPLOYMENT",
        "LOCAL_ECONOMIC_DEVELOPMENT",
        "LOCAL_GOVERNMENT_ADMINISTRATION",
        "NATURAL_RESOURCES",
        "PARKS_AND_PLAYGROUNDS",
        "PERSONS_WITH_DISABILITIES",
        "PUBLIC_ASSETS_AND_PROPERTY",
        "PUBLIC_FINANCE_AND_BUDGET",
        "PUBLIC_HEALTH_AND_SANITATION",
        "PUBLIC_SAFETY_AND_PEACE",
        "PUBLIC_UTILITIES",
        "REVENUE_CODE",
        "SENIOR_CITIZENS",
        "SOCIAL_WELFARE",
        "SPORTS",
        "TOURISM",
        "TRAFFIC_AND_TRANSPORTATION",
        "TRANSPARENCY_AND_ACCOUNTABILITY",
        "URBAN_POOR",
        "URBAN_PLANNING_AND_DEVELOPMENT",
        "WATER_RESOURCES_MANAGEMENT",
        "WASTE_MANAGEMENT",
    ]

    subCategories = [
        "ADMINISTRATIVE_ORDER",
        "ACKNOWLEDGEMENT",
        "AMENDMENT",
        "ANNUAL_INVESTMENT_PLAN",
        "AWARDS",
        "COMMENDATION",
        "BUDGET_APPROPRIATION",
        "EMERGENCY_DECLARATION",
        "GENERAL_ORDINANCE",
        "GENERAL_RESOLUTION",
        "IMPLEMENTING_RULES_AND_REGULATIONS",
        "REPEAL_OR_REVOCATION",
        "SUPPLEMENTAL_BUDGET",
    ]

    df = pd.read_csv("ordinances-1-cuid.csv")

    print(f"Loaded {len(df)} ordinances")

    output_file = "ordinances-2.csv"
    start_index = 0

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)

        if len(existing_df) == len(df) and existing_df["classifications"].notna().all():
            print("All ordinances already classified. Exiting.")
            return

        for i, row in existing_df.iterrows():
            if pd.isna(row["classifications"]):
                start_index = i
                break
        else:
            start_index = len(existing_df)

        print(f"Resuming from index {start_index}")

        # Copy data from existing file
        if "classifications" in existing_df.columns:
            df["classifications"] = existing_df["classifications"]

        if "subcategories" in existing_df.columns:
            df["subcategories"] = existing_df["subcategories"]

        if "confidence_score" in existing_df.columns:
            df["confidence_score"] = existing_df["confidence_score"]

        if "diagnostic_note" in existing_df.columns:
            df["diagnostic_note"] = existing_df["diagnostic_note"]

    else:
        # Create new columns
        df["classifications"] = None
        df["subcategories"] = None
        df["confidence_score"] = None
        df["diagnostic_note"] = None

    # Process ordinances in batches
    batch_size = 16  # Increased from 4 to 16

    titles_to_process = df["title"].iloc[start_index:].tolist()

    if not titles_to_process:
        print("No ordinances to process. Exiting.")
        return

    print(f"Processing {len(titles_to_process)} ordinances...")

    # Process in chunks and save periodically
    chunk_size = batch_size * 10  # Save every 10 batches

    for chunk_start in range(0, len(titles_to_process), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(titles_to_process))
        chunk_titles = titles_to_process[chunk_start:chunk_end]

        print(
            f"Processing chunk {chunk_start // chunk_size + 1}/{(len(titles_to_process) + chunk_size - 1) // chunk_size}"
        )

        classifications, subcategories_results, confidence_scores, diagnostic_notes = (
            classify_ordinances_batch(
                model,
                tokenizer,
                chunk_titles,
                categories,
                subCategories,
                "ordinance",
                device,
                batch_size,
            )
        )

        # Update DataFrame with chunk results
        chunk_start_idx = start_index + chunk_start
        chunk_end_idx = chunk_start_idx + len(classifications)

        df.loc[chunk_start_idx : chunk_end_idx - 1, "classifications"] = classifications
        df.loc[chunk_start_idx : chunk_end_idx - 1, "subcategories"] = (
            subcategories_results
        )
        df.loc[chunk_start_idx : chunk_end_idx - 1, "confidence_score"] = (
            confidence_scores
        )
        df.loc[chunk_start_idx : chunk_end_idx - 1, "diagnostic_note"] = (
            diagnostic_notes
        )

        # Save progress after each chunk
        df.to_csv(output_file, index=False)
        print(
            f"Progress saved: {chunk_end_idx - start_index}/{len(titles_to_process)} completed"
        )

        # Force garbage collection after each chunk
        torch.cuda.empty_cache()
        gc.collect()

    # Report category counts
    category_counts = {}
    for cls in df["classifications"].dropna():
        for category in cls.split(", "):
            category_name = category.split(" (")[0] if " (" in category else category
            category_counts[category_name] = category_counts.get(category_name, 0) + 1

    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    print("\n=== CATEGORY DISTRIBUTION (Nature) ===")
    for category, count in sorted_categories:
        print(f"{category}: {count} ordinances")

    # Report subcategory counts
    subcategory_counts = {}
    for subcat in df["subcategories"].dropna():
        subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    sorted_subcategories = sorted(
        subcategory_counts.items(), key=lambda x: x[1], reverse=True
    )
    print("\n=== SUBCATEGORY DISTRIBUTION (Type) ===")
    for subcategory, count in sorted_subcategories:
        print(f"{subcategory}: {count} ordinances")

    print("\nClassification complete!")


if __name__ == "__main__":
    main()
