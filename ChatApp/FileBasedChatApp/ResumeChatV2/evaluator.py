import json
import os
from dotenv import load_dotenv
import pandas as pd
from eval_utils import evaluate_retrieval_batch
from semantic_searcher_with_rerank import SemanticSearcherWithRerank


def evaluate(
    src_dir: str,
    eval_file: str,
    results_detailed_file: str,
    results_summary_file: str,
    semantic_searcher: SemanticSearcherWithRerank,
):
    results_path = os.path.join(src_dir, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # load the evaluation data
    with open(os.path.join(src_dir, eval_file), "r") as f:
        eval_data = json.load(f)

    (
        avg_precision,
        avg_recall,
        avg_mrr,
        avg_f1,
        precisions,
        recalls,
        f1s,
        mrrs,
        verdicts,
        all_correct_links,
        all_retrieved_links,
    ) = evaluate_retrieval_batch(eval_data, semantic_searcher)

    df = pd.DataFrame(
        {
            "question": [item["question"] for item in eval_data],
            "correct_links": all_correct_links,
            "retrieved_links": all_retrieved_links,
            "retrieval_precision": precisions,
            "retrieval_recall": recalls,
            "retrieval_mrr": mrrs,
            "retrieval_f1": f1s,
            "verdict": verdicts,
        }
    )
    df.to_csv(
        os.path.join(results_path, results_detailed_file),
        index=False,
    )

    with open(os.path.join(results_path, results_summary_file), "w") as f:
        json.dump(
            {
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1": avg_f1,
                "average_mrr": avg_mrr,
            },
            f,
            indent=2,
        )


load_dotenv(override=True)
src_dir = os.path.expanduser(
        "~/genAI/ChatApp/FileBasedChatApp/data"
    )
vector_db_dir = "faissResumeV2"
kb_dir = "resume"
embedding_model_name = os.getenv("HF_EMBEDDINGS_MODEL")
reranking_model_name = os.getenv("HF_ReRanker_MODEL")
retriever_top_k = 5
reranker_top_k = 2
ss = SemanticSearcherWithRerank(
        src_dir,
        kb_dir,
        vector_db_dir,
        embedding_model_name,
        reranking_model_name,
        retriever_top_k,
        reranker_top_k
    )

evaluate(
    src_dir=src_dir,
    eval_file="eval/resumev2_eval_data.json",
    results_detailed_file="results.csv",
    results_summary_file="summary.json",
    semantic_searcher=ss,
)