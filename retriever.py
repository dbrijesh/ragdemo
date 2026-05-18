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


def get_all_embeddings() -> dict:
    collection = _get_collection()
    return collection.get(include=["embeddings", "documents", "metadatas"])
