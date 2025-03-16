import json
import os
from dotenv import load_dotenv
import pandas as pd
from eval_utils import evaluate_retrieval_batch
from retriever import DataRetriever
from ingestor import DataIngestor


def evaluate(
    src_dir: str,
    eval_file: str,
    results_detailed_file: str,
    results_summary_file: str,
    retriever: DataRetriever,
    rerank: bool = False,
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
    ) = evaluate_retrieval_batch(eval_data, rerank, retriever)

    df = pd.DataFrame(
        {
            "question": [item["question"] for item in eval_data],
            "verdict": verdicts,
            "retrieval_precision": precisions,
            "retrieval_recall": recalls,
            "retrieval_mrr": mrrs,
            "retrieval_f1": f1s,
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
    "~/Documents/genai-training-pydanticai/data/semantic-search/project/version2"
)
vector_db_dir = "index"
kb_dir = "kb"
di = DataIngestor(src_dir, vector_db_dir, kb_dir)
di.ingest()
dr = DataRetriever(src_dir, vector_db_dir)

evaluate(
    src_dir=src_dir,
    eval_file="../eval/docs_evaluation_dataset.json",
    results_detailed_file="results.csv",
    results_summary_file="summary.json",
    retriever=dr,
)
evaluate(
    src_dir=src_dir,
    eval_file="../eval/docs_evaluation_dataset.json",
    results_detailed_file="results_rerank.csv",
    results_summary_file="summary_rerank.json",
    retriever=dr,
    rerank=True,
)