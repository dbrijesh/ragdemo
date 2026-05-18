import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from retriever import get_all_embeddings


def _run_tsne(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    n = len(embeddings)
    perplexity = min(30, max(5, n // 3))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def _load_data():
    raw = get_all_embeddings()
    embeddings = np.array(raw["embeddings"], dtype=np.float32)
    sources = [m["source"] for m in raw["metadatas"]]
    chunk_ids = [m["chunk_index"] for m in raw["metadatas"]]
    previews = [(d[:120] + "…") if len(d) > 120 else d for d in raw["documents"]]
    return embeddings, sources, chunk_ids, previews


def create_2d_figure():
    embeddings, sources, chunk_ids, previews = _load_data()
    coords = _run_tsne(embeddings, n_components=2)

    df = pd.DataFrame({
        "t-SNE 1":      coords[:, 0],
        "t-SNE 2":      coords[:, 1],
        "Document":     sources,
        "Chunk":        chunk_ids,
        "Text Preview": previews,
    })

    fig = px.scatter(
        df, x="t-SNE 1", y="t-SNE 2",
        color="Document",
        hover_name="Text Preview",
        hover_data={"Chunk": True, "t-SNE 1": False, "t-SNE 2": False, "Document": False},
        title="Ingested Chunks — 2D t-SNE",
        color_discrete_sequence=px.colors.qualitative.Bold,
        width=950, height=650,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title_text="Document", plot_bgcolor="#0e1117",
                      paper_bgcolor="#0e1117", font_color="#fafafa")
    return fig


def create_3d_figure():
    embeddings, sources, chunk_ids, previews = _load_data()
    coords = _run_tsne(embeddings, n_components=3)

    df = pd.DataFrame({
        "t-SNE 1":      coords[:, 0],
        "t-SNE 2":      coords[:, 1],
        "t-SNE 3":      coords[:, 2],
        "Document":     sources,
        "Chunk":        chunk_ids,
        "Text Preview": previews,
    })

    fig = px.scatter_3d(
        df, x="t-SNE 1", y="t-SNE 2", z="t-SNE 3",
        color="Document",
        hover_name="Text Preview",
        hover_data={"Chunk": True, "t-SNE 1": False, "t-SNE 2": False,
                    "t-SNE 3": False, "Document": False},
        title="Ingested Chunks — 3D t-SNE",
        color_discrete_sequence=px.colors.qualitative.Bold,
        width=950, height=720,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.85))
    fig.update_layout(legend_title_text="Document", paper_bgcolor="#0e1117",
                      font_color="#fafafa")
    return fig
