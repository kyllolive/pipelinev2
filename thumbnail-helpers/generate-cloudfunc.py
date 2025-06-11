import functions_framework
from google.cloud import storage
import fitz  # PyMuPDF
from PIL import Image
import io
import os


# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def generateDocThumbnail(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]
    bucket_name = data["bucket"]
    file_name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket_name}")
    print(f"File: {file_name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")

    # Check if file is in either ordinance/pdf/ or resolution/pdf/
    document_type = None
    if file_name.startswith("ordinance/pdf/"):
        document_type = "ordinance"
    elif file_name.startswith("resolution/pdf/"):
        document_type = "resolution"
    else:
        print(
            f"File {file_name} is not in ordinance/pdf/ or resolution/pdf/ folder, skipping"
        )
        return

    # Only process PDF files
    if not file_name.lower().endswith(".pdf"):
        print(f"File {file_name} is not a PDF, skipping")
        return

    # Only process finalize events (file creation/upload completion)
    if event_type != "google.cloud.storage.object.v1.finalized":
        print(f"Event type {event_type} is not finalized, skipping")
        return

    try:
        # Initialize storage client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        print(f"Processing {document_type} PDF: {file_name}")
        print(f"Downloading PDF: {file_name}")
        pdf_bytes = blob.download_as_bytes()

        # Convert first page of PDF to image
        print("Converting PDF to image...")

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            print("No pages found in PDF")
            doc.close()
            return

        # Get first page
        page = doc[0]

        # Render page to image (150 DPI)
        mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI scaling
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img_data = pix.tobytes("jpeg")
        image = Image.open(io.BytesIO(img_data))

        # Close the document
        doc.close()

        # Resize image to match generate.py logic
        print("Resizing image...")
        size = (300, 300)  # Fixed size like generate.py when size is specified
        image = image.resize(size, Image.LANCZOS)

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="WEBP", quality=80, optimize=True)
        img_byte_arr = img_byte_arr.getvalue()

        # Generate thumbnail filename based on document type
        # Extract filename without extension from the original path
        base_filename = os.path.basename(file_name)
        filename_without_ext = os.path.splitext(base_filename)[0]
        thumbnail_name = f"{document_type}/thumbnail/{filename_without_ext}.webp"

        # Upload thumbnail to GCS
        print(f"Uploading thumbnail: {thumbnail_name}")
        thumbnail_blob = bucket.blob(thumbnail_name)
        thumbnail_blob.upload_from_string(img_byte_arr, content_type="image/webp")

        # Set cache control for better performance
        thumbnail_blob.cache_control = "public, max-age=3600"
        thumbnail_blob.patch()

        print(f"Successfully generated thumbnail: {thumbnail_name}")

    except Exception as e:
        print(f"Error generating thumbnail for {file_name}: {str(e)}")
        # Clean up resources if they exist
        try:
            if "doc" in locals():
                doc.close()
        except:
            pass
        # Re-raise the exception to mark the function as failed
        raise
