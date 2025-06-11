import os
import argparse
import fitz
from PIL import Image
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_pdf(args):
    """Process a single PDF file and extract thumbnail"""
    pdf_path, output_dir, format, dpi, size, quality = args

    pdf_file = os.path.basename(pdf_path)

    try:
        # Open the PDF file with a timeout to prevent hanging on corrupt PDFs
        doc = fitz.open(pdf_path)

        if doc.page_count == 0:
            logger.warning(f"PDF has no pages: {pdf_file}")
            doc.close()
            return False

        # Get the first page
        page = doc[0]

        # Calculate zoom factor based on DPI (72 dpi is the base)
        zoom = dpi / 72

        # Create a pixmap (adjust the matrix for higher resolution)
        # Use RGB colorspace without alpha channel for smaller size
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Create output filename
        output_filename = os.path.splitext(pdf_file)[0] + f".{format}"
        output_path = os.path.join(output_dir, output_filename)

        # For small output sizes, use direct rendering to file instead of PIL
        if size is None and format.lower() in ["png", "jpg", "jpeg"]:
            pixmap.save(output_path)
        else:
            # Convert to PIL Image for resizing if needed
            img_data = pixmap.tobytes(
                "jpeg" if format.lower() in ["jpg", "jpeg"] else "png"
            )
            img = Image.open(io.BytesIO(img_data))

            # Resize if specified
            if size:
                img = img.resize(size, Image.LANCZOS)

            # Save with appropriate quality settings
            if format.lower() in ["jpg", "jpeg"]:
                img.save(
                    output_path, format=format.upper(), quality=quality, optimize=True
                )
            elif format.lower() == "webp":
                img.save(output_path, format=format.upper(), quality=quality, method=6)
            elif format.lower() == "png":
                img.save(output_path, format=format.upper(), optimize=True)
            else:
                img.save(output_path, format=format.upper())

        # Clean up
        doc.close()
        return True

    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {str(e)}")
        return False


def extract_first_page_thumbnails(
    input_dir,
    output_dir,
    format="png",
    dpi=150,
    size=None,
    quality=80,
    max_workers=None,
    batch_size=100,
):
    """
    Extracts the first page of each PDF using parallel processing for speed.

    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save thumbnail images
        format (str): Image format to save
        dpi (int): DPI resolution for conversion
        size (tuple): Optional size to resize the thumbnail
        quality (int): Quality for lossy formats
        max_workers (int): Maximum number of worker processes
        batch_size (int): Number of PDFs to process in each batch
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Get list of PDF files with full paths
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    # Determine the number of workers
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() + 4)

    total_files = len(pdf_files)
    logger.info(
        f"Found {total_files} PDF files. Processing with {max_workers} workers..."
    )

    # Prepare arguments for each task
    process_args = [
        (pdf_path, output_dir, format, dpi, size, quality) for pdf_path in pdf_files
    ]

    # Initialize counters
    completed = 0
    start_time = time.time()

    # Process in batches to avoid memory issues with very large directories
    for i in range(0, len(process_args), batch_size):
        batch = process_args[i : i + batch_size]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(process_pdf, batch):
                completed += 1 if result else 0

                # Print progress every 100 files or at the end
                if completed % 100 == 0 or completed == total_files:
                    elapsed = time.time() - start_time
                    files_per_second = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{total_files} ({files_per_second:.2f} files/second)"
                    )

    # Final report
    elapsed = time.time() - start_time
    files_per_second = total_files / elapsed if elapsed > 0 else 0
    logger.info(
        f"Processing complete! Processed {completed}/{total_files} files in {elapsed:.2f} seconds ({files_per_second:.2f} files/second)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract first pages from PDFs as thumbnail images"
    )
    parser.add_argument(
        "input_dir", help="Directory containing PDF files (will search recursively)"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save thumbnails (default: ./thumbnails)",
        default="./thumbnails",
    )
    parser.add_argument(
        "--format",
        help="Image format (default: webp)",
        default="webp",
        choices=["png", "jpg", "jpeg", "webp"],
    )
    parser.add_argument(
        "--dpi", help="DPI resolution (default: 150)", type=int, default=150
    )
    parser.add_argument("--width", help="Thumbnail width (optional)", type=int)
    parser.add_argument("--height", help="Thumbnail height (optional)", type=int)
    parser.add_argument(
        "--quality", help="JPEG/WebP quality (1-100, default: 80)", type=int, default=80
    )
    parser.add_argument(
        "--workers", help="Maximum number of worker processes (default: auto)", type=int
    )
    parser.add_argument(
        "--batch_size",
        help="Number of PDFs to process in each batch (default: 100)",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    # Determine size if width and height specified
    size = None
    if args.width and args.height:
        size = (args.width, args.height)

    # Start processing
    extract_first_page_thumbnails(
        args.input_dir,
        args.output_dir,
        format=args.format,
        dpi=args.dpi,
        size=size,
        quality=args.quality,
        max_workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
