import json
import pandas as pd
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
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
    context_str = "\n\n---\n\n".join(contexts)
    prompt = (
        "Answer the question using only the information in the context below. "
        "Be concise and accurate. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return llm.invoke(prompt).content


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
    samples = []
    per_question_rows = []

    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        if progress_callback:
            progress_callback(i, len(questions) + 1, f"Q{i+1}: {question[:60]}…")

        contexts = retrieve(question)
        answer = _generate_answer(question, contexts, llm)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        per_question_rows.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "num_contexts": len(contexts),
        })

    if progress_callback:
        progress_callback(len(questions), len(questions) + 1, "Running RAGAS metrics…")

    dataset = EvaluationDataset(samples=samples)
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    if progress_callback:
        progress_callback(len(questions) + 1, len(questions) + 1, "Complete.")

    def _agg(val):
        if isinstance(val, list):
            import numpy as np
            valid = [v for v in val if v is not None]
            return float(np.nanmean(valid)) if valid else 0.0
        return float(val)

    scores = {
        "faithfulness": _agg(result["faithfulness"]),
        "answer_relevancy": _agg(result["answer_relevancy"]),
        "context_precision": _agg(result["llm_context_precision_with_reference"]),
        "context_recall": _agg(result["context_recall"]),
    }

    detail_df = result.to_pandas()
    if "question" not in detail_df.columns:
        detail_df.insert(0, "question", [r["question"] for r in per_question_rows])

    return {"scores": scores, "detail_df": detail_df, "per_question": per_question_rows}
