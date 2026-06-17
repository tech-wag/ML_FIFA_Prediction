import os
from google.cloud import aiplatform

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_DISPLAY_NAME = "fifa-rf-model"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
MODEL_ARTIFACT_PATH = "model/fifa_model.pkl"


def upload_model_to_gcs(local_path, bucket_name, destination_path):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_path}")
    return f"gs://{bucket_name}/{destination_path}"


def deploy_vertex_model(gcs_model_uri, project_id=PROJECT_ID, location=LOCATION):
    aiplatform.init(project=project_id, location=location)

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=gcs_model_uri.rsplit("/", 1)[0],
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        artifact_uri_prefix=gcs_model_uri,
    )

    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )

    print(f"Deployed model to endpoint: {endpoint.resource_name}")
    return endpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload and deploy FIFA model to Vertex AI.")
    parser.add_argument("--local-model", default="fifa_model.pkl", help="Local path to the model pickle file")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--project", default=PROJECT_ID, help="GCP project ID")
    parser.add_argument("--location", default=LOCATION, help="Vertex AI location")
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("GCS bucket name must be provided via --bucket or GCS_BUCKET_NAME env var")
    if not args.project:
        raise ValueError("GCP project ID must be provided via --project or GOOGLE_CLOUD_PROJECT env var")

    gcs_uri = upload_model_to_gcs(args.local_model, args.bucket, MODEL_ARTIFACT_PATH)
    endpoint = deploy_vertex_model(gcs_uri, project_id=args.project, location=args.location)
    print(endpoint)
