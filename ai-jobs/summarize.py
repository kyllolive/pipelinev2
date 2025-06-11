from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import pandas as pd
import torch
import logging
import time
import gc
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def ensure_complete_sentence(summary):
    """Ensure the summary ends with a complete sentence."""
    # Find the last complete sentence (ends with period, exclamation, or question mark)
    sentences = re.split(r"([.!?]+)", summary)

    # Reconstruct up to the last complete sentence
    complete_summary = ""
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence and punct:
                complete_summary += sentence + punct + " "

    # If we found complete sentences, use that. Otherwise, return original
    if complete_summary.strip():
        return complete_summary.strip()
    else:
        # If no complete sentences found, try to end at a reasonable break point
        # Look for the last comma or semicolon
        for punct in [";", ","]:
            last_punct = summary.rfind(punct)
            if last_punct > len(summary) * 0.7:  # Only if it's in the latter part
                return summary[: last_punct + 1].strip()

        # If no good break point, return as is but add indication it's incomplete
        return summary.strip() + "..."


def summarize_batch_optimized(texts, model, tokenizer, device):
    """
    Optimized batch summarization with performance tracking.
    Focuses on substantive content: policies, regulations, and directives.
    """
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    batch_start_time = time.time()

    for text in texts:
        # Skip empty or very short texts
        if not isinstance(text, str) or len(text.strip()) < 50:
            all_results.append("")
            continue

        # Truncate very long texts to avoid OOM errors
        if len(text) > 2000:
            text = text[:2000] + "..."

        # Enhanced prompt for concise substantive summarization
        prompt = f"""You are a legal document analyst specializing in municipal ordinances. Your task is to extract and summarize only the most important substantive legal content that impacts citizens, businesses, or organizations.

        DOCUMENT TO ANALYZE:
        {text}

        ANALYSIS REQUIREMENTS:

        **INCLUDE (Most Important Content Only):**
        - Key policies, regulations, and legal requirements
        - Major prohibited activities and restrictions
        - Essential compliance obligations and procedures
        - Primary enforcement mechanisms and authority
        - Significant penalties, fines, and sanctions
        - Critical licensing or permit requirements
        - Important rights, obligations, and responsibilities
        - Key implementation timelines and effective dates
        - Essential definitions of legal terms

        **EXCLUDE (Administrative Content):**
        - Meeting attendance and procedural votes
        - Acknowledgments and ceremonial language
        - Administrative formalities and routine procedures
        - Introductory/background text without legal effect
        - Whereas clauses (unless they define scope)

        **OUTPUT REQUIREMENTS:**
        - Write a CONCISE narrative summary in 2-3 focused paragraphs
        - Prioritize the most impactful provisions over comprehensive coverage
        - Use clear, precise legal language in flowing prose
        - Focus on what citizens/businesses need to know most
        - Include specific amounts, dates, and requirements only if critically important
        - Maintain formal, analytical tone but be selective and concise

        **STRUCTURE:**
        1. First paragraph: Primary purpose, scope, and main regulations
        2. Second paragraph (if needed): Implementation details and effective dates

        Write a focused, concise analysis that captures the essential legal impact:"""

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
                max_new_tokens=200,  # Fixed to 200 tokens
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Slight penalty to avoid loops
                length_penalty=1.0,  # Neutral length penalty
                early_stopping=False,  # Don't stop early
            )

        # Track output tokens
        output_token_count = output.shape[1] - input_token_count
        total_output_tokens += output_token_count

        # Decode the generated tokens
        generated_tokens = output[0, input_ids["input_ids"].shape[1] :]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean up the summary
        summary = clean_summary(summary)

        # Check if summary was truncated and ensure it ends at a complete sentence
        if len(generated_tokens) >= 600 - 1:  # Close to max tokens
            logger.warning(
                f"Summary may be truncated (used {len(generated_tokens)}/600 tokens)"
            )
            summary = ensure_complete_sentence(summary)
        else:
            # Even if not truncated, ensure it ends properly
            summary = ensure_complete_sentence(summary)

        all_results.append(summary)

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
        "texts_processed": len(texts),
        "avg_tokens_per_text": total_tokens / len(texts) if texts else 0,
        "text_lengths": [len(text) if isinstance(text, str) else 0 for text in texts],
        "max_tokens_used": 1000,
    }

    return all_results, stats


def clean_summary(summary):
    """Clean and format the summary text."""
    # Remove excessive whitespace
    summary = re.sub(r"\s+", " ", summary).strip()

    # Remove common unwanted phrases
    unwanted_phrases = [
        "Here is a detailed summary:",
        "Summary:",
        "Based on the document:",
        "The document contains:",
    ]

    for phrase in unwanted_phrases:
        summary = summary.replace(phrase, "").strip()

    # Ensure bullet points are properly formatted
    summary = re.sub(r"^\s*[•·\-\*]\s*", "• ", summary, flags=re.MULTILINE)

    return summary


def process_summaries_batch(model, tokenizer, texts, device, batch_size=8):
    """Process texts in batches with performance monitoring."""
    all_summaries = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}: {len(batch_texts)} texts"
        )

        try:
            # Process batch
            batch_summaries, batch_stats = summarize_batch_optimized(
                batch_texts, model, tokenizer, device
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
            print(f"     • Avg tokens/text: {batch_stats['avg_tokens_per_text']:.1f}")

            # Show text length and token allocation distribution
            text_lengths = batch_stats["text_lengths"]

            if text_lengths:
                min_length = min(text_lengths)
                max_length = max(text_lengths)
                avg_length = sum(text_lengths) / len(text_lengths)

                print(f"  📝 Text Analysis:")
                print(
                    f"     • Text lengths: {min_length}-{max_length} chars (avg: {avg_length:.0f})"
                )

            # Store results
            all_summaries.extend(batch_summaries)

        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            # Add fallback values for the entire batch
            for _ in batch_texts:
                all_summaries.append("[Error during summarization]")

        # Memory cleanup
        if i % (batch_size * 5) == 0:  # Every 5 batches
            torch.cuda.empty_cache()
            gc.collect()

    return all_summaries


def main():
    try:
        # Set up device info
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # Read CSV file
        logger.info("Reading CSV file...")
        input_file = "resolutions-1-cuid.csv"
        if not os.path.exists(input_file):
            # Try alternative filename
            input_file = "resolutions-1-cuid.csv"

        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")

        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")

        # Model configuration - change to granite-3.3-2b-instruct for 3-4x speed boost
        model_path = (
            "ibm-granite/granite-3.3-2b-instruct"  # Default: slower but higher quality
        )
        # model_path = "ibm-granite/granite-3.3-2b-instruct"  # Alternative: much faster, good quality

        logger.info(f"Using model: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            use_cache=True,
            trust_remote_code=True,
            # H100 optimization: Enable 8-bit quantization for 2x speed boost
            # load_in_4bit=True,  # Uncomment for 4x speed, more quality trade-off (alternative to 8bit)
        )

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Prepare output file and check for existing progress
        output_file = "resolutions-2-summarized.csv"
        start_index = 0

        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)

            if len(existing_df) == len(df) and existing_df["summary"].notna().all():
                logger.info("All texts already summarized. Exiting.")
                return

            # Find where to resume
            for i, row in existing_df.iterrows():
                if pd.isna(row["summary"]):
                    start_index = i
                    break
            else:
                start_index = len(existing_df)

            logger.info(f"Resuming from index {start_index}")

            # Copy existing data
            if "summary" in existing_df.columns:
                df["summary"] = existing_df["summary"]
        else:
            # Create new summary column
            df["summary"] = None

        # Filter texts to process
        texts_to_process = []
        valid_indices = []

        for idx in range(start_index, len(df)):
            text = df.iloc[idx]["detected_text"]
            if isinstance(text, str) and len(text.strip()) >= 50:
                texts_to_process.append(text)
                valid_indices.append(idx)

        if not texts_to_process:
            logger.info("No texts to process. Exiting.")
            return

        logger.info(f"Processing {len(texts_to_process)} texts...")

        # Process in chunks and save periodically
        batch_size = 16  # Increased for H100 from 8 to 16
        chunk_size = batch_size * 5  # Save every 5 batches instead of 10

        for chunk_start in range(0, len(texts_to_process), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts_to_process))
            chunk_texts = texts_to_process[chunk_start:chunk_end]
            chunk_indices = valid_indices[chunk_start:chunk_end]

            logger.info(
                f"Processing chunk {chunk_start // chunk_size + 1}/{(len(texts_to_process) + chunk_size - 1) // chunk_size}"
            )

            # Process chunk
            summaries = process_summaries_batch(
                model, tokenizer, chunk_texts, device, batch_size
            )

            # Update DataFrame with chunk results
            for idx, summary in zip(chunk_indices, summaries):
                df.loc[idx, "summary"] = summary

            # Save progress after each chunk
            df.to_csv(output_file, index=False)
            logger.info(
                f"Progress saved: {chunk_end}/{len(texts_to_process)} completed"
            )

            # Force garbage collection after each chunk
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Summarization complete! Results saved to {output_file}")

        # Report statistics
        completed_summaries = df["summary"].notna().sum()
        logger.info(f"Total summaries generated: {completed_summaries}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
