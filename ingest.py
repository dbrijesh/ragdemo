import os
import json
import boto3
import chromadb
import config


def _embed_single(text: str, client) -> list[float]:
    body = json.dumps({
        "inputText": text[:8000],
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


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + config.CHUNK_SIZE])
        chunks.append(chunk)
        i += config.CHUNK_SIZE - config.CHUNK_OVERLAP
        if i + config.CHUNK_OVERLAP >= len(words):
            break
    return chunks


def ingest_documents(progress_callback=None, log_callback=None) -> dict:
    def log(msg):
        if log_callback:
            log_callback(msg)

    log("=" * 60)
    log("  RAGAS INGESTION PIPELINE STARTED")
    log("=" * 60)
    log(f"  Model  : {config.BEDROCK_EMBED_MODEL}")
    log(f"  Profile: {config.AWS_PROFILE}  |  Region: {config.AWS_REGION}")
    log(f"  Chunk size: {config.CHUNK_SIZE} words  |  Overlap: {config.CHUNK_OVERLAP}")
    log(f"  Embedding dims: {config.EMBED_DIMENSIONS}")
    log("")

    session = boto3.Session(
        **({'profile_name': config.AWS_PROFILE} if config.AWS_PROFILE else {}),
        region_name=config.AWS_REGION,
    )
    bedrock = session.client("bedrock-runtime")

    log("  [ChromaDB] Initialising persistent store...")
    chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        chroma.delete_collection(config.COLLECTION_NAME)
        log("  [ChromaDB] Dropped existing collection")
    except Exception:
        pass
    collection = chroma.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log(f"  [ChromaDB] Collection '{config.COLLECTION_NAME}' created (cosine similarity)")
    log("")

    doc_files = sorted(f for f in os.listdir(config.DOCUMENTS_PATH) if f.endswith(".txt"))
    log(f"  Found {len(doc_files)} documents to process")
    log("")

    doc_stats = {}

    for idx, doc_file in enumerate(doc_files):
        if progress_callback:
            progress_callback(idx, len(doc_files), f"Ingesting {doc_file}…")

        path = os.path.join(config.DOCUMENTS_PATH, doc_file)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        word_count = len(text.split())
        doc_name = doc_file.replace(".txt", "")

        log(f"  {'─'*55}")
        log(f"  DOCUMENT [{idx+1}/{len(doc_files)}]: {doc_file}")
        log(f"  {'─'*55}")
        log(f"  >> Loaded  : {word_count:,} words  |  {len(text):,} characters")

        # ── chunking ──────────────────────────────────────────
        log(f"  >> Chunking: splitting into ~{config.CHUNK_SIZE}-word pieces...")
        chunks = _chunk_text(text)
        log(f"  >> Created : {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            preview = chunk[:70].replace("\n", " ")
            log(f"       Chunk {i+1:02d}: \"{preview}...\"")

        # ── embedding ─────────────────────────────────────────
        log(f"")
        log(f"  >> Embedding {len(chunks)} chunks via Bedrock Titan...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            log(f"       [{i+1}/{len(chunks)}] Calling InvokeModel... ", )
            emb = _embed_single(chunk, bedrock)
            embeddings.append(emb)
            log(f"       [{i+1}/{len(chunks)}] Done -> {config.EMBED_DIMENSIONS}D vector  "
                f"(norm={round(sum(v**2 for v in emb[:5])**.5, 4)})")

        # ── storing ───────────────────────────────────────────
        log(f"")
        log(f"  >> Storing in ChromaDB...")
        ids       = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        log(f"  >> Stored  : {len(chunks)} vectors  |  IDs: {ids[0]} … {ids[-1]}")
        log(f"  >> ChromaDB total: {collection.count()} chunks so far")
        log("")

        doc_stats[doc_name] = len(chunks)

    if progress_callback:
        progress_callback(len(doc_files), len(doc_files), "Done.")

    total = sum(doc_stats.values())
    log("=" * 60)
    log(f"  INGESTION COMPLETE")
    log(f"  Total chunks stored : {total}")
    log(f"  Documents processed : {len(doc_files)}")
    for name, count in doc_stats.items():
        log(f"    {name:<35} {count} chunks")
    log("=" * 60)

    return {"total_chunks": total, "documents": doc_stats}


def collection_exists() -> bool:
    try:
        chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
        return chroma.get_collection(config.COLLECTION_NAME).count() > 0
    except Exception:
        return False
