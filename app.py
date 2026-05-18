import json, os, re
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

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #050a14 0%, #0a1628 60%, #0d1f35 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04080f 0%, #06101e 100%);
    border-right: 1px solid #1a3a5c;
}
[data-testid="stHeader"] { background: transparent; }

/* ── typography ── */
h1 { color: #00d4ff !important; letter-spacing: -0.5px; }
h2 { color: #5bb8d4 !important; }
h3 { color: #7fc8d8 !important; }
p, li, label { color: #c8dde8; }

/* ── sidebar radio ── */
[data-testid="stSidebar"] label { color: #8ab4c8 !important; font-size: 13px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #5a8aa8; font-size: 12px; }

/* ── primary buttons ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #0066cc 0%, #0044aa 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 0 18px rgba(0, 100, 200, 0.35) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 28px rgba(0, 140, 255, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── expanders ── */
details summary {
    background: #0a1e32 !important;
    border-radius: 8px !important;
    color: #7fc8d8 !important;
    border: 1px solid #1a3a5c !important;
}
details[open] { border: 1px solid #1a4a6b !important; border-radius: 8px !important; }

/* ── dataframes ── */
[data-testid="stDataFrame"] { border: 1px solid #1a3a5c; border-radius: 8px; }

/* ── code blocks ── */
[data-testid="stCode"] > div {
    background: #040c18 !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
}

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #5a8aa8;
    border-radius: 6px 6px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00d4ff;
    border-bottom: 2px solid #00d4ff;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    background: #07111e !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── constants & state ─────────────────────────────────────────────────────────
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
        st.button(label or f"Next: {STEPS[idx+1].strip()} →",
                  type="primary", on_click=_go_next,
                  use_container_width=True, key=f"nxt_{idx}")


def _score_color(v: float) -> str:
    if np.isnan(v):    return "#5a7a9a"
    if v >= 0.80:      return "#00e676"
    if v >= 0.60:      return "#ffd740"
    return "#ff5252"


def _fmt(v) -> str:
    if v is None: return "N/A"
    try:
        f = float(v)
        return "N/A" if np.isnan(f) else f"{f:.3f}"
    except Exception:
        return "N/A"


def _agg(val) -> float:
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else float("nan")
    try:    return float(val)
    except: return float("nan")


def _metric_card(label: str, value: str, description: str = "") -> str:
    try:
        fv  = float(value)
        clr = _score_color(fv)
        pct = f"{fv*100:.1f}"
        bar = f'<div style="background:{clr};width:{pct}%;height:4px;border-radius:4px;box-shadow:0 0 8px {clr};"></div>'
    except Exception:
        clr = "#5a7a9a"; pct = "0"; bar = ""

    desc_html = f'<div style="color:#4a6a82;font-size:11px;margin-top:6px;">{description}</div>' if description else ""
    return f"""
    <div style="background:linear-gradient(135deg,#0d2137 0%,#081623 100%);
                border:1px solid #1a3a5c;border-radius:12px;padding:18px 20px;
                margin:6px 0;box-shadow:0 2px 20px rgba(0,100,200,0.08);">
      <div style="color:#5a8aa8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;">{label}</div>
      <div style="font-size:2rem;font-weight:700;color:{clr};margin:6px 0;font-family:monospace;">{value}</div>
      <div style="background:#0d2035;border-radius:4px;height:4px;">{bar}</div>
      {desc_html}
    </div>"""


def _rank_table(chunks: list[dict], score_key: str, score_label: str,
                highlight_up_to: int = 0, pre_ranks: dict = None) -> str:
    rows = ""
    for i, c in enumerate(chunks):
        is_used  = highlight_up_to > 0 and i < highlight_up_to
        bg       = "rgba(0,100,200,0.12)" if is_used else "transparent"
        border_l = "3px solid #0088ff" if is_used else "3px solid transparent"
        score    = c.get(score_key, "—")
        score_s  = f"{score:.4f}" if isinstance(score, float) else str(score)

        delta = ""
        if pre_ranks and score_key == "llm_score":
            old = pre_ranks.get(id(c), i + 1)
            diff = old - (i + 1)
            if diff > 0:   delta = f'<span style="color:#00e676;font-size:11px;"> ▲{diff}</span>'
            elif diff < 0: delta = f'<span style="color:#ff5252;font-size:11px;"> ▼{abs(diff)}</span>'

        used_badge = '<span style="color:#00d4ff;font-size:10px;font-weight:700;"> ✦USED</span>' if is_used else ""
        preview    = c["text"][:90].replace("<","&lt;").replace("\n"," ") + "…"
        source     = c["source"].replace("_", " ")

        rows += f"""
        <tr style="background:{bg};border-left:{border_l};">
          <td style="padding:8px 10px;color:#5a8aa8;font-weight:700;">#{i+1}</td>
          <td style="padding:8px 10px;color:#00d4ff;font-size:12px;">{source}{used_badge}</td>
          <td style="padding:8px 10px;color:#ffd740;font-family:monospace;white-space:nowrap;">{score_s}{delta}</td>
          <td style="padding:8px 10px;color:#8ab4c8;font-size:12px;">{preview}</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto;margin:8px 0;">
      <div style="color:#4a7a9a;font-size:11px;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;">{score_label}</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead><tr style="border-bottom:1px solid #1a3a5c;">
          <th style="padding:6px 10px;color:#3a6a8a;text-align:left;">Rank</th>
          <th style="padding:6px 10px;color:#3a6a8a;text-align:left;">Source</th>
          <th style="padding:6px 10px;color:#3a6a8a;text-align:left;">{score_label.split()[0]}</th>
          <th style="padding:6px 10px;color:#3a6a8a;text-align:left;">Preview</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


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


def _llm_rerank(question: str, chunks: list[dict], top_k: int) -> tuple[list[dict], list[dict]]:
    """Score all chunks with one LLM call and return (pre_rank, post_rank[:top_k])."""
    llm = get_llm()
    passages = "\n\n".join(
        f"[{i+1}] {c['text'][:350]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"Rate each passage's relevance to the query (1=irrelevant, 10=perfect match).\n"
        f"Reply ONLY with comma-separated integers, one per passage.\n\n"
        f"Query: {question}\n\n{passages}\n\nScores:"
    )
    try:
        raw    = llm.invoke(prompt).content.strip()
        scores = [min(10, max(1, int(x))) for x in re.findall(r"\d+", raw)][:len(chunks)]
        while len(scores) < len(chunks):
            scores.append(5)
    except Exception:
        scores = [5] * len(chunks)

    pre = []
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        c = dict(chunk)
        c["pre_rank"]  = i + 1
        c["llm_score"] = score
        pre.append(c)

    post = sorted(pre, key=lambda x: x["llm_score"], reverse=True)
    for i, c in enumerate(post):
        c["post_rank"] = i + 1

    return pre, post[:top_k]


def rag_answer(question: str):
    from retriever import retrieve_with_scores
    all_chunks = retrieve_with_scores(question, k=config.TOP_K * 2)
    pre_rank, post_rank = _llm_rerank(question, all_chunks, config.TOP_K)

    contexts = [c["text"] for c in post_rank]
    ctx      = "\n\n---\n\n".join(contexts)
    prompt   = (
        "Answer the question using only the information in the context below. "
        "Be concise and accurate.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = get_llm().invoke(prompt).content
    return answer, contexts, pre_rank, post_rank


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#00d4ff;font-size:1.3rem;font-weight:700;'
                'letter-spacing:-0.3px;margin-bottom:2px;">RAGAS Demo</div>', unsafe_allow_html=True)
    st.caption("RAG Evaluation · Amazon Bedrock")
    st.divider()
    st.radio("Demo Steps", STEPS, key="step")
    st.divider()

    def _sb_status(label, ok):
        dot = '<span style="color:#00e676;">●</span>' if ok else '<span style="color:#ff5252;">●</span>'
        st.markdown(f'{dot} <span style="color:#8ab4c8;font-size:12px;">{label}</span>',
                    unsafe_allow_html=True)

    _sb_status("Knowledge base ingested",  st.session_state["ingested"])
    _sb_status("Full evaluation complete", bool(st.session_state["eval_result"]))
    _sb_status("Chat eval complete",       bool(st.session_state["chat_eval"]))

    if st.session_state["chat_history"]:
        n = len(st.session_state["chat_history"])
        st.markdown(f'<span style="color:#5a8aa8;font-size:12px;">💬 {n} chat message(s)</span>',
                    unsafe_allow_html=True)
    st.divider()
    st.markdown(f'<span style="color:#2a5a7a;font-size:11px;">Model: `{config.BEDROCK_LLM_MODEL.split(".")[-1]}`</span>',
                unsafe_allow_html=True)
    st.markdown(f'<span style="color:#2a5a7a;font-size:11px;">Profile: `{config.AWS_PROFILE or "IAM role"}`</span>',
                unsafe_allow_html=True)

step = st.session_state["step"]


def _step_header(title: str, subtitle: str):
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,rgba(0,212,255,0.08) 0%,transparent 100%);
                border-left:3px solid #00d4ff;border-radius:0 8px 8px 0;
                padding:16px 20px;margin-bottom:20px;">
      <div style="font-size:1.6rem;font-weight:700;color:#00d4ff;">{title}</div>
      <div style="color:#5a8aa8;font-size:14px;margin-top:4px;">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
if step == STEPS[0]:
    _step_header("Step 1 — Ingest Knowledge Base",
                 "Chunk documents → embed with Titan Text Embeddings v2 → store in ChromaDB")

    doc_files = sorted(f for f in os.listdir(config.DOCUMENTS_PATH) if f.endswith(".txt"))
    col_docs, col_run = st.columns([3, 2], gap="large")

    with col_docs:
        st.markdown('<div style="color:#5bb8d4;font-weight:600;margin-bottom:10px;">'
                    f'Knowledge Base — {len(doc_files)} Documents</div>', unsafe_allow_html=True)

        TOPIC_COLORS = {
            "ai_overview": "#ff6b6b", "deep_learning": "#ff6b6b",
            "transformers": "#4ecdc4", "attention_mechanisms": "#4ecdc4", "nlp_fundamentals": "#4ecdc4",
            "rag_systems": "#45b7d1", "vector_databases": "#45b7d1", "dense_retrieval": "#45b7d1",
            "llm_evaluation": "#96ceb4", "ragas_deep_dive": "#96ceb4",
        }
        for doc in doc_files:
            path = os.path.join(config.DOCUMENTS_PATH, doc)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            key = doc.replace(".txt", "")
            clr = TOPIC_COLORS.get(key, "#7fa8c9")
            wc  = len(content.split())
            title = key.replace("_", " ").title()
            with st.expander(f"  {title} — {wc:,} words"):
                st.caption(f"Source: {doc}")
                st.text((content[:450] + "…") if len(content) > 450 else content)

    with col_run:
        st.markdown('<div style="color:#5bb8d4;font-weight:600;margin-bottom:10px;">Ingestion Pipeline</div>',
                    unsafe_allow_html=True)

        if st.session_state["ingested"]:
            st.markdown('<div style="background:rgba(0,230,118,0.08);border:1px solid #00e676;'
                        'border-radius:8px;padding:10px 14px;color:#00e676;font-size:13px;">'
                        '✓ Knowledge base is ready. Re-ingest to refresh.</div>', unsafe_allow_html=True)
            st.markdown("")

        if st.button("Ingest All Documents", type="primary", use_container_width=True):
            log_box = st.empty()
            prog    = st.progress(0.0)
            logs    = []

            def add_log(msg):
                logs.append(msg)
                log_box.code("\n".join(logs), language="")

            try:
                from ingest import ingest_documents
                stats = ingest_documents(
                    progress_callback=lambda c,t,_: prog.progress(c/max(t,1)),
                    log_callback=add_log
                )
                st.session_state["ingested"]    = True
                st.session_state["ingest_logs"] = logs
                prog.progress(1.0)
                st.markdown(f'<div style="background:rgba(0,230,118,0.08);border:1px solid #00e676;'
                            f'border-radius:8px;padding:12px 16px;color:#00e676;margin-top:12px;">'
                            f'✓ Done — <b>{stats["total_chunks"]}</b> chunks across '
                            f'<b>{len(stats["documents"])}</b> documents</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Error: {exc}")

        if st.session_state["ingest_logs"]:
            with st.expander("Ingestion log"):
                st.code("\n".join(st.session_state["ingest_logs"]), language="")

    st.divider()
    if st.session_state["ingested"]:
        _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[1]:
    _step_header("Step 2 — Test Questions",
                 "Pre-built Q&A pairs with ground-truth answers for full 4-metric RAGAS evaluation")

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
    _step_header("Step 3 — Chat with RAG",
                 "Ask anything · LLM reranks retrieved chunks · add expected answers to unlock all 4 metrics")

    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    # ── conversation history ──────────────────────────────────────────────
    for i, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.markdown(msg["question"])

        with st.chat_message("assistant"):
            st.markdown(msg["answer"])

            # reranking display
            pre  = msg.get("pre_rank",  [])
            post = msg.get("post_rank", [])
            if pre:
                pre_id_to_rank = {id(c): c.get("pre_rank", j+1) for j, c in enumerate(pre)}
                with st.expander(f"Context retrieval — {len(pre)} fetched → top {len(post)} after rerank"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            _rank_table(pre, "similarity", "Similarity Score (vector)"),
                            unsafe_allow_html=True)
                    with c2:
                        st.markdown(
                            _rank_table(post + [c for c in pre if c not in post],
                                        "llm_score", "LLM Relevance Score (1-10)",
                                        highlight_up_to=len(post)),
                            unsafe_allow_html=True)

            # optional reference
            ref = msg.get("reference", "")
            new_ref = st.text_area("Expected answer (optional — enables Context Precision & Recall)",
                                   value=ref, key=f"ref_{i}", height=60)
            if new_ref != ref:
                st.session_state["chat_history"][i]["reference"] = new_ref

    # ── input ─────────────────────────────────────────────────────────────
    user_q = st.chat_input("Ask a question about the knowledge base…")
    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving → Reranking → Generating…"):
                try:
                    answer, contexts, pre_rank, post_rank = rag_answer(user_q)
                    st.markdown(answer)

                    pre_id_to_rank = {id(c): c.get("pre_rank", j+1) for j, c in enumerate(pre_rank)}
                    with st.expander(f"Context retrieval — {len(pre_rank)} fetched → top {len(post_rank)} after rerank"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                _rank_table(pre_rank, "similarity", "Similarity Score (vector)"),
                                unsafe_allow_html=True)
                        with c2:
                            all_post = post_rank + [c for c in pre_rank if c not in post_rank]
                            st.markdown(
                                _rank_table(all_post, "llm_score", "LLM Relevance Score (1-10)",
                                            highlight_up_to=len(post_rank)),
                                unsafe_allow_html=True)

                    st.session_state["chat_history"].append({
                        "question":  user_q,
                        "answer":    answer,
                        "contexts":  contexts,
                        "pre_rank":  pre_rank,
                        "post_rank": post_rank,
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
        n     = len(st.session_state["chat_history"])
        n_ref = sum(1 for m in st.session_state["chat_history"] if m.get("reference","").strip())
        if n:
            st.markdown(f'<div style="background:rgba(0,212,255,0.06);border:1px solid #1a4a6b;'
                        f'border-radius:8px;padding:8px 14px;color:#5a8aa8;font-size:13px;text-align:center;">'
                        f'💬 {n} message(s) &nbsp;·&nbsp; {n_ref} with expected answers</div>',
                        unsafe_allow_html=True)

    st.divider()
    if st.session_state["chat_history"]:
        _next_btn("Next: Evaluate →")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[3]:
    _step_header("Step 4 — RAGAS Evaluation",
                 "Full test-set (4 metrics with custom faithfulness) · or evaluate your chat session")

    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    tab_full, tab_chat = st.tabs(["Full Test-Set  (4 metrics)", "Chat Session"])

    with tab_full:
        st.markdown("Runs **10 pre-built questions** through the pipeline and scores with: "
                    "**Faithfulness · Answer Relevancy · Context Precision · Context Recall**")
        if st.button("Run Full Evaluation", type="primary", use_container_width=True, key="run_full"):
            prog = st.progress(0.0)
            stat = st.empty()
            with st.spinner("Evaluating — may take a few minutes…"):
                try:
                    from evaluate import run_evaluation
                    result = run_evaluation(
                        progress_callback=lambda c,t,m: (prog.progress(c/max(t,1)), stat.text(m)))
                    st.session_state["eval_result"] = result
                    prog.progress(1.0)
                    stat.text("Complete.")
                    st.success("Full evaluation done — see Metrics step.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        if st.session_state["eval_result"]:
            scores = st.session_state["eval_result"]["scores"]
            DESCS  = {"faithfulness":"No hallucination","answer_relevancy":"On-topic answers",
                      "context_precision":"Retrieval precision","context_recall":"Retrieval completeness"}
            LABELS = {"faithfulness":"Faithfulness","answer_relevancy":"Answer Relevancy",
                      "context_precision":"Context Precision","context_recall":"Context Recall"}
            cols   = st.columns(4)
            for col, (k, v) in zip(cols, scores.items()):
                col.markdown(_metric_card(LABELS.get(k,k), _fmt(v), DESCS.get(k,"")),
                             unsafe_allow_html=True)

    with tab_chat:
        history = st.session_state["chat_history"]
        if not history:
            st.info("Have a conversation in Step 3 first.")
        else:
            has_refs = any(m.get("reference","").strip() for m in history)
            if has_refs:
                st.success(f"Expected answers found → **all 4 metrics** will run.")
            else:
                st.info("No expected answers → **Faithfulness + Answer Relevancy** only. "
                        "Add expected answers in Step 3 to unlock all 4 metrics.")

            st.dataframe(
                pd.DataFrame([{"Q": m["question"], "A": m["answer"][:80]+"…",
                               "Ref?": "✅" if m.get("reference","").strip() else "—"}
                              for m in history]),
                use_container_width=True)

            if st.button("Evaluate Chat Session", type="primary", use_container_width=True, key="run_chat"):
                prog2 = st.progress(0.0)
                stat2 = st.empty()
                with st.spinner("Evaluating…"):
                    try:
                        from langchain_aws import ChatBedrock, BedrockEmbeddings
                        from ragas import EvaluationDataset, evaluate as ragas_eval
                        from ragas.dataset_schema import SingleTurnSample
                        from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
                        from ragas.llms import LangchainLLMWrapper
                        from ragas.embeddings import LangchainEmbeddingsWrapper
                        from evaluate import compute_faithfulness

                        pk     = {"credentials_profile_name": config.AWS_PROFILE} if config.AWS_PROFILE else {}
                        llm_lc = ChatBedrock(model_id=config.BEDROCK_LLM_MODEL,
                                             region_name=config.AWS_REGION,
                                             model_kwargs={"max_tokens":512,"temperature":0}, **pk)
                        emb_lc = BedrockEmbeddings(model_id=config.BEDROCK_EMBED_MODEL,
                                                   region_name=config.AWS_REGION, **pk)

                        samples, faith_scores = [], []
                        for i, m in enumerate(history):
                            prog2.progress((i+1)/len(history))
                            stat2.text(f"Sample {i+1}/{len(history)}…")
                            ref = m.get("reference","").strip() or None
                            samples.append(SingleTurnSample(
                                user_input=m["question"], response=m["answer"],
                                retrieved_contexts=m["contexts"], reference=ref))
                            faith_scores.append(compute_faithfulness(m["answer"], m["contexts"], llm_lc))

                        stat2.text("Running RAGAS…")
                        metrics = [ResponseRelevancy()]
                        if has_refs:
                            metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]

                        result = ragas_eval(
                            dataset=EvaluationDataset(samples=samples),
                            metrics=metrics,
                            llm=LangchainLLMWrapper(llm_lc),
                            embeddings=LangchainEmbeddingsWrapper(emb_lc),
                        )

                        valid_f = [s for s in faith_scores if not np.isnan(s)]
                        chat_scores = {
                            "faithfulness":    float(np.mean(valid_f)) if valid_f else float("nan"),
                            "answer_relevancy": _agg(result["answer_relevancy"]),
                        }
                        if has_refs:
                            chat_scores["context_precision"] = _agg(result["llm_context_precision_with_reference"])
                            chat_scores["context_recall"]    = _agg(result["context_recall"])

                        st.session_state["chat_eval"] = {"scores": chat_scores, "detail_df": result.to_pandas()}
                        prog2.progress(1.0); stat2.text("Done.")
                        st.success("Chat evaluation complete!")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

            if st.session_state["chat_eval"]:
                LABELS = {"faithfulness":"Faithfulness","answer_relevancy":"Answer Relevancy",
                          "context_precision":"Context Precision","context_recall":"Context Recall"}
                scores = st.session_state["chat_eval"]["scores"]
                cols   = st.columns(len(scores))
                for col, (k, v) in zip(cols, scores.items()):
                    col.markdown(_metric_card(LABELS.get(k,k), _fmt(v)), unsafe_allow_html=True)

    st.divider()
    if st.session_state["eval_result"] or st.session_state["chat_eval"]:
        _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[4]:
    _step_header("Step 5 — Metrics Dashboard",
                 "Aggregate scores · per-question breakdown · comparison across test-set and chat")

    if not st.session_state["eval_result"] and not st.session_state["chat_eval"]:
        st.info("Run an evaluation in Step 4 first.")
        st.stop()

    LABELS = {"faithfulness":"Faithfulness","answer_relevancy":"Answer Relevancy",
              "context_precision":"Context Precision","context_recall":"Context Recall"}
    DESCS  = {"faithfulness":"Hallucination check","answer_relevancy":"On-topic answers",
              "context_precision":"Retrieval precision","context_recall":"Retrieval completeness"}

    if st.session_state["eval_result"]:
        st.markdown("#### Test-Set Scores")
        scores = st.session_state["eval_result"]["scores"]
        cols   = st.columns(4)
        for col, (k, v) in zip(cols, scores.items()):
            col.markdown(_metric_card(LABELS.get(k,k), _fmt(v), DESCS.get(k,"")), unsafe_allow_html=True)

    if st.session_state["chat_eval"]:
        st.markdown("#### Chat Session Scores")
        chat_scores = st.session_state["chat_eval"]["scores"]
        cols2 = st.columns(len(chat_scores))
        for col, (k, v) in zip(cols2, chat_scores.items()):
            col.markdown(_metric_card(LABELS.get(k,k), _fmt(v), DESCS.get(k,"")), unsafe_allow_html=True)

    st.divider()

    # comparison bar
    bar_rows = []
    if st.session_state["eval_result"]:
        for k, v in st.session_state["eval_result"]["scores"].items():
            try:
                fv = float(v)
                if not np.isnan(fv):
                    bar_rows.append({"Metric": LABELS.get(k,k), "Score": fv, "Source": "Test Set"})
            except Exception: pass
    if st.session_state["chat_eval"]:
        for k, v in st.session_state["chat_eval"]["scores"].items():
            try:
                fv = float(v)
                if not np.isnan(fv):
                    bar_rows.append({"Metric": LABELS.get(k,k), "Score": fv, "Source": "Chat"})
            except Exception: pass

    if bar_rows:
        fig_bar = px.bar(
            pd.DataFrame(bar_rows), x="Metric", y="Score", color="Source",
            barmode="group", text_auto=".3f", range_y=[0, 1.12],
            title="RAGAS Score Comparison",
            color_discrete_sequence=["#00d4ff", "#ffd740"],
        )
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#c8dde8", title_font_color="#00d4ff",
                               legend=dict(bgcolor="rgba(0,0,0,0)"),
                               xaxis=dict(gridcolor="#1a3a5c"), yaxis=dict(gridcolor="#1a3a5c"))
        fig_bar.update_traces(textposition="outside", textfont_color="#c8dde8")
        st.plotly_chart(fig_bar, use_container_width=True)

    if st.session_state["eval_result"]:
        st.markdown("#### Per-Question Breakdown")
        detail_df = st.session_state["eval_result"]["detail_df"]
        st.dataframe(detail_df, use_container_width=True, height=380)

        metric_cols = [c for c in detail_df.columns
                       if c in ["faithfulness","answer_relevancy",
                                 "llm_context_precision_with_reference","context_recall"]]
        if metric_cols and "question" in detail_df.columns:
            heat_df = detail_df[["question"] + metric_cols].copy()
            heat_df["question"] = heat_df["question"].str[:45] + "…"
            heat_df = heat_df.set_index("question")
            heat_df.columns = [LABELS.get(c,c) for c in heat_df.columns]
            fig_heat = px.imshow(heat_df.T, color_continuous_scale="RdYlGn",
                                 zmin=0, zmax=1, aspect="auto", title="Score Heatmap")
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8dde8",
                                    title_font_color="#00d4ff")
            st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    _next_btn()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VISUALISE
# ══════════════════════════════════════════════════════════════════════════════
elif step == STEPS[5]:
    _step_header("Step 6 — Embedding Visualisation",
                 "t-SNE reduces 1024D vectors to 2D/3D · hover any point to read the chunk")

    if not st.session_state["ingested"]:
        st.warning("Ingest the knowledge base first (Step 1).")
        st.stop()

    st.markdown("""
    <div style="background:rgba(0,212,255,0.05);border:1px solid #1a4a6b;border-radius:10px;
                padding:14px 18px;margin-bottom:16px;color:#5a8aa8;font-size:13px;line-height:1.6;">
      Chunks from the same document cluster together. Related topics form nearby super-clusters:<br>
      <span style="color:#ff6b6b;">■</span> AI / Deep Learning &nbsp;
      <span style="color:#4ecdc4;">■</span> NLP / Transformers &nbsp;
      <span style="color:#45b7d1;">■</span> Retrieval / RAG &nbsp;
      <span style="color:#96ceb4;">■</span> Evaluation
    </div>""", unsafe_allow_html=True)

    c2d, c3d = st.columns(2)
    with c2d:
        if st.button("Generate 2D t-SNE", type="primary", use_container_width=True):
            with st.spinner("Computing t-SNE (2D)…"):
                try:
                    from visualize import create_2d_figure
                    st.session_state["fig_2d"] = create_2d_figure()
                except Exception as exc:
                    st.error(f"Error: {exc}")
    with c3d:
        if st.button("Generate 3D t-SNE", type="primary", use_container_width=True):
            with st.spinner("Computing t-SNE (3D)…"):
                try:
                    from visualize import create_3d_figure
                    st.session_state["fig_3d"] = create_3d_figure()
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if st.session_state["fig_2d"]:
        st.plotly_chart(st.session_state["fig_2d"], use_container_width=True)
    if st.session_state["fig_3d"]:
        st.plotly_chart(st.session_state["fig_3d"], use_container_width=True)
