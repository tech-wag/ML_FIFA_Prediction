import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from google.cloud import aiplatform, storage
except ImportError:
    aiplatform = None
    storage = None

DEFAULT_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_INDEX_PATH = "rag_index.jsonl"
DEFAULT_EMBEDDING_DIM = 384


def _normalize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("id", doc.get("link", ""))),
        "title": doc.get("title", ""),
        "summary": doc.get("summary", ""),
        "link": doc.get("link", ""),
        "published": doc.get("published", ""),
    }


def document_text(doc: Dict[str, Any]) -> str:
    return f"{doc.get('title', '')} {doc.get('summary', '')}".strip()


def build_embeddings(
    documents: Iterable[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> List[np.ndarray]:
    model = SentenceTransformer(model_name)
    texts = [document_text(_normalize_document(doc)) for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def save_index(documents: List[Dict[str, Any]], embeddings: np.ndarray, path: str = DEFAULT_INDEX_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc, emb in zip(documents, embeddings):
            entry = _normalize_document(doc).copy()
            entry["embedding"] = emb.tolist()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved RAG index to {path}")


def load_index(path: str = DEFAULT_INDEX_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def search_local_index(
    query: str,
    index_documents: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Dict[str, Any]]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    embeddings = np.vstack([np.array(doc["embedding"], dtype=np.float32) for doc in index_documents])
    scores = embeddings @ query_embedding
    top_indices = list(np.argsort(scores)[::-1][:top_k])
    return [index_documents[i] for i in top_indices]


def init_vertex(project: Optional[str] = None, location: str = "us-central1") -> None:
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is required for Vertex Matching Engine integration.")
    aiplatform.init(project=project, location=location)


def upload_file_to_gcs(local_path: str, bucket_name: str, destination_path: str) -> str:
    if storage is None:
        raise RuntimeError("google-cloud-storage is required to upload files to GCS.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{bucket_name}/{destination_path}"
    print(f"Uploaded {local_path} to {gcs_uri}")
    return gcs_uri


def create_vertex_matching_engine_index(
    display_name: str,
    gcs_index_uri: str,
    project: Optional[str] = None,
    location: str = "us-central1",
    dimension: int = DEFAULT_EMBEDDING_DIM,
) -> Any:
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is required for Vertex Matching Engine integration.")

    init_vertex(project=project, location=location)
    index = aiplatform.MatchingEngineIndex.create(
        display_name=display_name,
        contents_delta_uri=gcs_index_uri,
        metadata={"dimension": dimension},
    )
    print(f"Created Vertex Matching Engine index: {index.resource_name}")
    return index


def query_vertex_matching_engine(
    index_resource_name: str,
    query: str,
    project: Optional[str] = None,
    location: str = "us-central1",
    top_k: int = 5,
) -> Any:
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is required for Vertex Matching Engine integration.")

    init_vertex(project=project, location=location)
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()

    index = aiplatform.MatchingEngineIndex(index_resource_name)
    response = index.match(instances=[{"embedding": query_embedding}], return_max_results=top_k)
    return response


def build_production_rag_index(articles: List[Dict[str, Any]], index_path: str = DEFAULT_INDEX_PATH) -> None:
    embeddings = build_embeddings(articles)
    save_index(articles, embeddings, path=index_path)


if __name__ == "__main__":
    import argparse
    from rag_enrich import fetch_google_news

    parser = argparse.ArgumentParser(description="Build and query a production-grade RAG index.")
    parser.add_argument("--build-index", action="store_true", help="Build a local RAG index from news articles")
    parser.add_argument("--query", type=str, help="Query the local RAG index")
    parser.add_argument("--output", default=DEFAULT_INDEX_PATH, help="Local index path")
    parser.add_argument("--team", default="Brazil", help="Team name to query or build for")
    parser.add_argument("--project", help="GCP project for Vertex Matching Engine")
    parser.add_argument("--bucket", help="GCS bucket for uploading local index")
    parser.add_argument("--gcs-destination", default="rag_index/rag_index.jsonl", help="GCS destination path")
    parser.add_argument("--use-vertex", action="store_true", help="Build and upload index to Vertex Matching Engine")
    args = parser.parse_args()

    if args.build_index:
        docs = fetch_google_news(f"{args.team} football", max_items=50)
        build_production_rag_index(docs, index_path=args.output)

        if args.use_vertex:
            if not args.bucket or not args.project:
                raise ValueError("--bucket and --project are required for Vertex upload")
            gcs_uri = upload_file_to_gcs(args.output, args.bucket, args.gcs_destination)
            print(f"Local index built and uploaded to: {gcs_uri}")
    elif args.query:
        docs = load_index(args.output)
        results = search_local_index(args.query, docs, top_k=5)
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()
