import json
import re
import numpy as np
import pandas as pd
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import config
from retriever import retrieve


def _load_test_data() -> tuple[list[str], list[str]]:
    questions, ground_truths = [], []
    with open(config.TEST_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                questions.append(item["question"])
                ground_truths.append(item["ground_truth"])
    return questions, ground_truths


def _generate_answer(question: str, contexts: list[str], llm) -> str:
    ctx = "\n\n---\n\n".join(contexts)
    prompt = (
        "Answer the question using only the information in the context below. "
        "Be concise and accurate. If the answer is not in the context, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llm.invoke(prompt).content


def compute_faithfulness(answer: str, contexts: list[str], llm) -> float:
    """
    Custom faithfulness using two direct LLM calls:
      1. Extract atomic factual claims from the answer.
      2. Batch-verify each claim against the retrieved context.
    Returns a float in [0, 1] or NaN if parsing fails.
    """
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1: extract claims
    claims_raw = llm.invoke(
        "List every distinct factual claim in the answer below as a numbered list. "
        "One claim per line. Be specific and brief.\n\n"
        f"Answer: {answer}\n\nClaims:"
    ).content

    claims = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in claims_raw.splitlines()
        if re.match(r"^\s*\d+", line) and line.strip()
    ]

    if not claims:
        return float("nan")

    # Step 2: batch-verify
    claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verdict_raw = llm.invoke(
        "For each numbered claim, state whether the context supports it.\n"
        "Reply ONLY in the format:  1:yes  2:no  3:yes  ...\n\n"
        f"Context:\n{ctx}\n\nClaims:\n{claims_text}\n\nVerdict:"
    ).content.lower()

    yes_count = len(re.findall(r":\s*yes", verdict_raw))
    no_count  = len(re.findall(r":\s*no",  verdict_raw))
    total     = yes_count + no_count

    return round(yes_count / total, 4) if total > 0 else float("nan")


def _agg(val) -> float:
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def run_evaluation(progress_callback=None) -> dict:
    profile_kwarg = {"credentials_profile_name": config.AWS_PROFILE} if config.AWS_PROFILE else {}
    llm = ChatBedrock(
        model_id=config.BEDROCK_LLM_MODEL,
        region_name=config.AWS_REGION,
        model_kwargs={"max_tokens": 512, "temperature": 0},
        **profile_kwarg,
    )
    embeddings = BedrockEmbeddings(
        model_id=config.BEDROCK_EMBED_MODEL,
        region_name=config.AWS_REGION,
        **profile_kwarg,
    )

    questions, ground_truths = _load_test_data()
    samples            = []
    per_question_rows  = []
    faithfulness_scores = []

    total_steps = len(questions) + 1

    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        if progress_callback:
            progress_callback(i, total_steps, f"Q{i+1}/{len(questions)}: {question[:55]}…")

        contexts = retrieve(question)
        answer   = _generate_answer(question, contexts, llm)
        faith    = compute_faithfulness(answer, contexts, llm)
        faithfulness_scores.append(faith)

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))
        per_question_rows.append({
            "question":     question,
            "answer":       answer,
            "ground_truth": ground_truth,
            "faithfulness": faith,
        })

    if progress_callback:
        progress_callback(len(questions), total_steps, "Running RAGAS metrics…")

    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    if progress_callback:
        progress_callback(total_steps, total_steps, "Complete.")

    valid_faith = [s for s in faithfulness_scores if not np.isnan(s)]
    scores = {
        "faithfulness":      float(np.mean(valid_faith)) if valid_faith else float("nan"),
        "answer_relevancy":  _agg(result["answer_relevancy"]),
        "context_precision": _agg(result["llm_context_precision_with_reference"]),
        "context_recall":    _agg(result["context_recall"]),
    }

    detail_df = result.to_pandas()
    detail_df.insert(0, "question",     [r["question"]     for r in per_question_rows])
    detail_df.insert(1, "faithfulness", [r["faithfulness"] for r in per_question_rows])

    return {"scores": scores, "detail_df": detail_df, "per_question": per_question_rows}
