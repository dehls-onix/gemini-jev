#!/usr/bin/env bash
set -e

project_id="search-ahmed"
region="us-central1"
service_name="gemini-jev-decision-workbench"
image="us-central1-docker.pkg.dev/${project_id}/cloud-run/gemini-jev:latest"

echo "building image: ${image}"
gcloud builds submit --tag "${image}" --project "${project_id}" --quiet

echo "deploying service: ${service_name}"
gcloud run deploy "${service_name}" \
  --image "${image}" \
  --project "${project_id}" \
  --region "${region}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "GCP_PROJECT_ID=${project_id},BIGQUERY_DATASET=jev_laya_eval,JEV_API_KEY=${JEV_API_KEY:-apikey_placeholder},HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY:-hf_placeholder}" \
  --quiet

echo "service deployed:"
gcloud run services describe "${service_name}" --project "${project_id}" --region "${region}" --format 'value(status.url)'

