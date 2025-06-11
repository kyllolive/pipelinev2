# Generate Thumbnail when uploading to bucket


## Deploying a new version? Use this:

gcloud functions deploy generateDocThumbnail \
  --runtime python311 \
  --source ./tools \
  --entry-point generateDocThumbnail \
  --trigger-event google.storage.object.finalize \
  --trigger-resource cityhall-raw \
  --memory 512MB \
  --timeout 120s \
  --region=asia-east2