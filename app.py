import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from ingest import collection_exists

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGAS Evaluation Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    "1  Ingest Documents",
    "2  Test Data",
    "3  Chat with RAG",
    "4  Evaluate",
    "5  Metrics",
    "6  Visualise",
]

for key, default in [
    ("step",         STEPS[0]),
    ("ingested",     collection_exists()),
    ("chat_history", []),
    ("eval_result",  None),
    ("chat_eval",    None),
    ("fig_2d",       None),
    ("fig_3d",       None),
    ("ingest_logs",  []),
]:
    st.session_state.setdefault(key, default)


# ── helpers ───────────────────────────────────────────────────────────────────
def _go_next():
    idx = STEPS.index(st.session_state["step"])
    if idx < len(STEPS) - 1:
        st.session_state["step"] = STEPS[idx + 1]


def _next_btn(label=None):
    idx = STEPS.index(st.session_state["step"])
    if idx < len(STEPS) - 1:
        st.button(
            label or f"Next: {STEPS[idx + 1].strip()} →",
            type="primary", on_click=_go_next,
            use_container_width=True, key=f"nxt_{idx}",
        )


def _fmt(v):
    """Format a metric score, showing N/A for None/NaN."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
        return "N/A" if np.isnan(f) else f"{f:.3f}"
    except Exception:
        return "N/A"


def _agg(val):
    """Aggregate a list of per-sample scores into a single float."""
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else float("nan")
    return float(val) if val is not None else float("nan")


# ── cached LLM ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    from langchain_aws import ChatBedrock
    kw = {"credentials_profile_name": config.AWS_PROFILE} if config.AWS_PROFILE else {}
    return ChatBedrock(
        model_id=config.BEDROCK_LLM_MODEL,
        region_name=config.AWS_REGION,
        model_kwargs={"max_tokens": 512, "temperature": 0.1},
        **kw,
    )


def rag_answer(question: str):
    from retriever import retrieve
    contexts = retrieve(question)
    ctx      = "\n\n---\n\n".join(contexts)
    prompt   = (
        "Answer the question using only the information in the context below. "
        "Be concise and accurate. If the answer is not in the context, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    return get_llm().invoke(prompt).content, contexts


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAGAS Demo")
    st.caption("RAG Evaluation · Amazon Bedrock")
    st.divider()
    st.radio("Demo Steps", STEPS, key="step")
    st.divider()
    st.markdown("**Status**")
    st.write("Knowledge base:", "✅ Ready" if st.session_state["ingested"] else "❌ Not ingested")
    st.write("Chat messages :", len(st.session_state["chat_history"]))
    st.write("Full eval     :", "✅ Done" if st.session_state["eval_result"] else "—")
    st.write("Chat eval     :", "✅ Done" if st.session_state["chat_eval"]   else "—")
    st.divider()
    st.caption(f"LLM  : `{config.BEDROCK_LLM_MODEL}`")
    st.caption(f"Embed: `{config.BEDROCK_EMBED_MODEL}`")
    st.caption(f"Profile: `{config.AWS_PROFILE or 'IAM role'}`")

step = st.session_state["step"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
if step == STEPS[0]:
    st.title("Step 1 — Ingest Knowledge Base")
    st.markdown(
        "Documents are **chunked → embedded** via Bedrock Titan Text Embeddings v2 → "
        "stored in **ChromaDB** with cosine similarity index."
    )

    doc_files = sorted(f for f in os.listdir(config.DOCUMENTS_PATH) if f.endswith(".txt"))
    col_docs, col_run = st.columns([3, 2], gap="large")

    with col_docs:
        st.subheader(f"Documents ({len(doc_files)})")
        for doc in doc_files:
            path = os.path.join(config.DOCUMENTS_PATH, doc)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            wc = len(content.split())
            with st.expander(f"{doc.replace('.txt','').replace('_',' ').title()}  —  {wc:,} words"):
                st.text((content[:500] + "…") if len(content) > 500 else content)

    with col_run:
        st.subheader("Run Ingestion")
        if st.session_state["ingested"]:
            st.success("Already ingested. Re-ingest to refresh.")

        if st.button("Ingest All Documents", type="primary", use_container_width=True):
            log_box = st.empty()
            prog    = st.progress(0.0)
            logs    = []

            def add_log(msg):
                logs.append(msg)
                log_box.code("\n".join(logs), language="")

            def on_prog(cur, tot, _msg):
                prog.progress(cur / max(tot, 1))

            try:
                from ingest import ingest_documents
                stats = ingest_documents(progress_callback=on_prog, log_callback=add_log)
                st.session_state["ingested"]    = True
                st.session_state["ingest_logs"] = logs
                prog.progress(1.0)
                st.success(f"Done — {stats['total_chunks']} chunks across {len(stats['documents'])} docs.")
            except Exception as exc:
                st.error(f"Error: {exc}")

        if st.session_state["ingest_logs"]:
            with st.expander("Show ingestion log", expanded=False):
                st.code("\n".join(st.session_state["ingest_logs"]), language="")

    st.divider()
    if st.session_state["ingested"]:
        _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[1]:
    st.title("Step 2 — Test Questions")
    st.markdown(
        "Pre-built **question / ground-truth** pairs used for full RAGAS evaluation "
        "(all 4 metrics including Context Precision and Context Recall)."
    )
    if os.path.exists(config.TEST_DATA_PATH):
        items = []
        with open(config.TEST_DATA_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        df = pd.DataFrame(items)
        st.caption(f"{len(df)} questions · `{config.TEST_DATA_PATH}`")
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.warning("Test data file not found.")
    st.divider()
    _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHAT WITH RAG
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[2]:
    st.title("Step 3 — Chat with RAG")
    st.markdown(
        "Ask anything about the knowledge base. Optionally provide an **expected answer** "
        "to enable all 4 RAGAS metrics during chat evaluation (otherwise 2 metrics run)."
    )

    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    # ── display history ───────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            c1, c2 = st.columns([2, 1])
            with c1:
                with st.expander(f"Retrieved contexts ({len(msg['contexts'])})"):
                    for j, ctx in enumerate(msg["contexts"], 1):
                        st.markdown(f"**Chunk {j}:** {ctx}")
            with c2:
                ref = msg.get("reference", "")
                new_ref = st.text_area(
                    "Expected answer (optional)",
                    value=ref,
                    key=f"ref_{i}",
                    height=80,
                    help="Provide this to enable Context Precision + Recall during evaluation.",
                )
                if new_ref != ref:
                    st.session_state["chat_history"][i]["reference"] = new_ref

    # ── input ─────────────────────────────────────────────────────────────
    user_q = st.chat_input("Ask a question about the knowledge base…")
    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    answer, contexts = rag_answer(user_q)
                    st.markdown(answer)
                    with st.expander(f"Retrieved contexts ({len(contexts)})"):
                        for j, ctx in enumerate(contexts, 1):
                            st.markdown(f"**Chunk {j}:** {ctx}")
                    st.session_state["chat_history"].append({
                        "question":  user_q,
                        "answer":    answer,
                        "contexts":  contexts,
                        "reference": "",
                    })
                except Exception as exc:
                    st.error(f"Error: {exc}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["chat_eval"]    = None
            st.rerun()
    with c2:
        n = len(st.session_state["chat_history"])
        n_ref = sum(1 for m in st.session_state["chat_history"] if m.get("reference","").strip())
        if n:
            st.info(f"{n} message(s) · {n_ref} with expected answers")

    st.divider()
    if st.session_state["chat_history"]:
        _next_btn("Next: Evaluate →")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[3]:
    st.title("Step 4 — RAGAS Evaluation")

    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    tab_full, tab_chat = st.tabs(["Full Test-Set Eval  (4 metrics)", "Chat Session Eval"])

    # ── full eval ─────────────────────────────────────────────────────────
    with tab_full:
        st.markdown(
            "Runs all **10 pre-built questions** and scores with: "
            "Faithfulness · Answer Relevancy · Context Precision · Context Recall"
        )
        if st.button("Run Full Evaluation", type="primary", use_container_width=True, key="run_full"):
            prog = st.progress(0.0)
            stat = st.empty()

            def on_fp(cur, tot, msg):
                prog.progress(cur / max(tot, 1))
                stat.text(msg)

            with st.spinner("Evaluating…"):
                try:
                    from evaluate import run_evaluation
                    result = run_evaluation(progress_callback=on_fp)
                    st.session_state["eval_result"] = result
                    prog.progress(1.0)
                    stat.text("Done.")
                    st.success("Full evaluation complete — see Metrics step.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        if st.session_state["eval_result"]:
            scores = st.session_state["eval_result"]["scores"]
            LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
                      "context_precision": "Context Precision", "context_recall": "Context Recall"}
            cols = st.columns(4)
            for col, (k, v) in zip(cols, scores.items()):
                col.metric(LABELS.get(k, k), _fmt(v))

    # ── chat eval ─────────────────────────────────────────────────────────
    with tab_chat:
        history = st.session_state["chat_history"]
        if not history:
            st.info("Have a conversation in Step 3 first.")
        else:
            has_refs = any(m.get("reference", "").strip() for m in history)
            if has_refs:
                st.success(
                    f"{sum(1 for m in history if m.get('reference','').strip())} / {len(history)} "
                    "messages have expected answers → **all 4 metrics** will run."
                )
            else:
                st.info(
                    "No expected answers provided → running **Faithfulness + Answer Relevancy** only. "
                    "Add expected answers in Step 3 to enable Context Precision + Recall."
                )

            rows = [{"Q": m["question"], "A": m["answer"][:90] + "…", "Ref?": "✅" if m.get("reference","").strip() else "—"}
                    for m in history]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if st.button("Evaluate Chat Session", type="primary", use_container_width=True, key="run_chat"):
                prog2 = st.progress(0.0)
                stat2 = st.empty()
                with st.spinner("Evaluating chat session…"):
                    try:
                        from langchain_aws import ChatBedrock, BedrockEmbeddings
                        from ragas import EvaluationDataset, evaluate as ragas_eval
                        from ragas.dataset_schema import SingleTurnSample
                        from ragas.metrics import (Faithfulness, ResponseRelevancy,
                                                   LLMContextPrecisionWithReference, LLMContextRecall)
                        from ragas.llms import LangchainLLMWrapper
                        from ragas.embeddings import LangchainEmbeddingsWrapper

                        pk = {"credentials_profile_name": config.AWS_PROFILE} if config.AWS_PROFILE else {}
                        llm_lc = ChatBedrock(model_id=config.BEDROCK_LLM_MODEL,
                                             region_name=config.AWS_REGION,
                                             model_kwargs={"max_tokens": 512, "temperature": 0}, **pk)
                        emb_lc = BedrockEmbeddings(model_id=config.BEDROCK_EMBED_MODEL,
                                                   region_name=config.AWS_REGION, **pk)

                        samples = []
                        for i, m in enumerate(history):
                            prog2.progress((i + 1) / len(history))
                            stat2.text(f"Preparing {i+1}/{len(history)}…")
                            ref = m.get("reference", "").strip() or None
                            samples.append(SingleTurnSample(
                                user_input=m["question"],
                                response=m["answer"],
                                retrieved_contexts=m["contexts"],
                                reference=ref,
                            ))

                        stat2.text("Running RAGAS metrics…")
                        use_full = has_refs
                        metrics = [Faithfulness(), ResponseRelevancy()]
                        if use_full:
                            metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]

                        result = ragas_eval(
                            dataset=EvaluationDataset(samples=samples),
                            metrics=metrics,
                            llm=LangchainLLMWrapper(llm_lc),
                            embeddings=LangchainEmbeddingsWrapper(emb_lc),
                        )

                        chat_scores = {
                            "faithfulness":     _agg(result["faithfulness"]),
                            "answer_relevancy": _agg(result["answer_relevancy"]),
                        }
                        if use_full:
                            chat_scores["context_precision"] = _agg(result["llm_context_precision_with_reference"])
                            chat_scores["context_recall"]    = _agg(result["context_recall"])

                        st.session_state["chat_eval"] = {
                            "scores":    chat_scores,
                            "detail_df": result.to_pandas(),
                        }
                        prog2.progress(1.0)
                        stat2.text("Done.")
                        st.success("Chat evaluation complete!")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

            if st.session_state["chat_eval"]:
                LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
                          "context_precision": "Context Precision", "context_recall": "Context Recall"}
                scores = st.session_state["chat_eval"]["scores"]
                cols   = st.columns(len(scores))
                for col, (k, v) in zip(cols, scores.items()):
                    col.metric(LABELS.get(k, k), _fmt(v))

    st.divider()
    if st.session_state["eval_result"] or st.session_state["chat_eval"]:
        _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[4]:
    st.title("Step 5 — Metrics")

    if not st.session_state["eval_result"] and not st.session_state["chat_eval"]:
        st.info("Run an evaluation in Step 4 first.")
        st.stop()

    LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
              "context_precision": "Context Precision", "context_recall": "Context Recall"}

    if st.session_state["eval_result"]:
        st.subheader("Full Test-Set — Aggregate Scores")
        scores = st.session_state["eval_result"]["scores"]
        cols   = st.columns(4)
        for col, (k, v) in zip(cols, scores.items()):
            col.metric(LABELS.get(k, k), _fmt(v))

    if st.session_state["chat_eval"]:
        st.subheader("Chat Session — Aggregate Scores")
        chat_scores = st.session_state["chat_eval"]["scores"]
        cols2 = st.columns(len(chat_scores))
        for col, (k, v) in zip(cols2, chat_scores.items()):
            col.metric(LABELS.get(k, k), _fmt(v))

    st.divider()

    # ── comparison bar chart ───────────────────────────────────────────────
    bar_rows = []
    if st.session_state["eval_result"]:
        for k, v in st.session_state["eval_result"]["scores"].items():
            if not np.isnan(float(v)) if v is not None else False:
                bar_rows.append({"Metric": LABELS.get(k, k), "Score": float(v), "Source": "Test Set"})
    if st.session_state["chat_eval"]:
        for k, v in st.session_state["chat_eval"]["scores"].items():
            try:
                fv = float(v)
                if not np.isnan(fv):
                    bar_rows.append({"Metric": LABELS.get(k, k), "Score": fv, "Source": "Chat"})
            except Exception:
                pass

    if bar_rows:
        fig_bar = px.bar(
            pd.DataFrame(bar_rows), x="Metric", y="Score",
            color="Source", barmode="group",
            text_auto=".3f", range_y=[0, 1.12],
            title="RAGAS Scores",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_bar.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa")
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── per-question heatmap ───────────────────────────────────────────────
    if st.session_state["eval_result"]:
        st.subheader("Per-Question Detail")
        detail_df = st.session_state["eval_result"]["detail_df"]
        st.dataframe(detail_df, use_container_width=True, height=380)

        metric_cols = [c for c in detail_df.columns if c in
                       ["faithfulness", "answer_relevancy",
                        "llm_context_precision_with_reference", "context_recall"]]
        if metric_cols and "question" in detail_df.columns:
            heat_df = detail_df[["question"] + metric_cols].copy()
            heat_df["question"] = heat_df["question"].str[:45] + "…"
            heat_df = heat_df.set_index("question")
            heat_df.columns = [LABELS.get(c, c) for c in heat_df.columns]
            fig_heat = px.imshow(heat_df.T, color_continuous_scale="RdYlGn",
                                 zmin=0, zmax=1, aspect="auto", title="Score Heatmap")
            fig_heat.update_layout(paper_bgcolor="#0e1117", font_color="#fafafa")
            st.plotly_chart(fig_heat, use_container_width=True)

    if st.session_state["chat_eval"]:
        st.subheader("Per-Message Detail (Chat)")
        st.dataframe(st.session_state["chat_eval"]["detail_df"], use_container_width=True)

    st.divider()
    _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VISUALISE
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[5]:
    st.title("Step 6 — Embedding Visualisation (t-SNE)")
    st.markdown(
        "Every ingested chunk is plotted in **2D / 3D** after t-SNE dimensionality reduction. "
        "Chunks from the same document cluster together; related topics form nearby clusters. "
        "Hover a point to read its text."
    )
    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    c2d, c3d = st.columns(2)
    with c2d:
        if st.button("Generate 2D t-SNE", type="primary", use_container_width=True):
            with st.spinner("Computing 2D t-SNE…"):
                try:
                    from visualize import create_2d_figure
                    st.session_state["fig_2d"] = create_2d_figure()
                except Exception as exc:
                    st.error(f"Error: {exc}")
    with c3d:
        if st.button("Generate 3D t-SNE", type="primary", use_container_width=True):
            with st.spinner("Computing 3D t-SNE…"):
                try:
                    from visualize import create_3d_figure
                    st.session_state["fig_3d"] = create_3d_figure()
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if st.session_state["fig_2d"]:
        st.plotly_chart(st.session_state["fig_2d"], use_container_width=True)
    if st.session_state["fig_3d"]:
        st.plotly_chart(st.session_state["fig_3d"], use_container_width=True)
