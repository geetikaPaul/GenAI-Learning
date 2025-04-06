import json
import os
from dotenv import load_dotenv
import pandas as pd
from eval_utils import evaluate_retrieval_batch
from semantic_search_with_rerank import SemanticSearch
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate, RunConfig
from langchain_groq import ChatGroq
from rag_with_CoT import rag_get_response
from tqdm import tqdm

load_dotenv(override=True)

def generate_llm_responses(
    src_dir: str,
    eval_file: str,
    ss: SemanticSearch):
        print(">>> Begin: Generating responses for evaluation questions...")
        with open(os.path.join(src_dir, eval_file), "r") as f:
            eval_data = json.load(f)

        results = []
        for item in tqdm(eval_data, desc="Generating Answers Using RAG"):
            response = rag_get_response(item["user_input"])
            retrieved_contexts = ss.retrieveContent(item["user_input"])
            tmp = {
                "id": item["id"],
                "user_input": item["user_input"],
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": item["reference"],
            }
            results.append(tmp)

        with open(os.path.join(src_dir, eval_file), "w") as f:
            json.dump(results, f, indent=2)
        print(">>> End: Generating responses for evaluation questions...")

def evaluate_generator(
    src_dir: str,
    eval_file: str):
    # test_data = SingleTurnSample(
    #     user_input="summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    #     response="The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    #     reference="The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter.",
    #     retrieved_contexts = ["The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."]
    # )
    # samples = []
    # samples.append(test_data)
    #evaluation_dataset = EvaluationDataset(samples=samples)
    
    with open(os.path.join(src_dir, eval_file), "r") as f:
            eval_data = json.load(f)

        # build evaluation dataset
    samples = []
    for item in eval_data:
            sample = SingleTurnSample(
                user_input=item["user_input"],
                reference_contexts=item["retrieved_contexts"],
                response=item["response"],
                reference=item["reference"],
            )
            samples.append(sample)
    evaluation_dataset = EvaluationDataset(samples=samples)

    llm = ChatGroq(
            model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY")
        )
    evaluator_llm = LangchainLLMWrapper(llm)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    # response_relevancy_metric = ResponseRelevancy(
    #         llm=evaluator_llm, embeddings=evaluator_embeddings
    #     )

    # evaluate metrics
    result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                faithfulness_metric
            ],
            run_config=RunConfig(max_workers=4, max_wait=60),
            llm=evaluator_llm,
            show_progress=True,
        )
    print(result)
    
def evaluate_retriever(
    src_dir: str,
    eval_file: str,
    results_detailed_file: str,
    results_summary_file: str,
    semantic_searcher: SemanticSearch,
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
            "question": [item["user_input"] for item in eval_data],
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
src_dir = os.path.expanduser("~/genAI/Chatapp/DbBasedChatApp/data")
kb_dir = "metadata"
vector_db_dir = "index"
embedding_model_name = os.getenv("HF_EMBEDDINGS_MODEL")
reranking_model_name = os.getenv("HF_ReRanker_MODEL")
retriever_top_k = 10
reranker_top_k = 5
ss = SemanticSearch(
        src_dir,
        kb_dir,
        vector_db_dir,
        embedding_model_name,
        reranking_model_name,
        retriever_top_k,
        reranker_top_k
    )

generate_llm_responses(src_dir=src_dir, eval_file="eval/chinook_metrics_whole_eval.json", ss=ss)
#evaluate_generator(src_dir=src_dir, eval_file="eval/chinook_metrics_whole_eval.json")
# evaluate_retriever(
#     src_dir=src_dir,
#     eval_file="eval/chinook_eval.json",
#     results_detailed_file="results.csv",
#     results_summary_file="summary.json",
#     semantic_searcher=ss,
# )