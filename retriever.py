import json
import boto3
import chromadb
import config


def _embed_query(query: str) -> list[float]:
    session = boto3.Session(
        **({'profile_name': config.AWS_PROFILE} if config.AWS_PROFILE else {}),
        region_name=config.AWS_REGION,
    )
    client = session.client("bedrock-runtime")
    body = json.dumps({
        "inputText": query[:8000],
        "dimensions": config.EMBED_DIMENSIONS,
        "normalize": True,
    })
    response = client.invoke_model(
        modelId=config.BEDROCK_EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def _get_collection():
    chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return chroma.get_collection(config.COLLECTION_NAME)


def retrieve(query: str, k: int = config.TOP_K) -> list[str]:
    embedding = _embed_query(query)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents"],
    )
    return results["documents"][0]


def retrieve_with_scores(query: str, k: int = config.TOP_K * 2) -> list[dict]:
    """Return chunks with similarity scores for reranking display."""
    embedding = _embed_query(query)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0=identical, 2=opposite → convert to similarity [0,1]
        chunks.append({
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity":  round(1 - dist / 2, 4),
        })
    return chunks


def get_all_embeddings() -> dict:
    collection = _get_collection()
    return collection.get(include=["embeddings", "documents", "metadatas"])
