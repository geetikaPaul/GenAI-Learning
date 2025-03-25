from tqdm import tqdm

def calculate_mrr(retrieved_chunks: list[str], correct_chunks: list[str]) -> float:
    for i, chunk in enumerate(retrieved_chunks, 1):
        if any(correct_chunk in chunk for correct_chunk in correct_chunks):
            return 1 / i
    return 0


def evaluate_retrieval(retrieved_chunks: list[str], correct_chunks: list[str]):
    true_positives = len([1 for retrieved in retrieved_chunks for correct in correct_chunks if correct in retrieved])
    precision = true_positives / len(retrieved_chunks) if retrieved_chunks else 0
    recall = true_positives / len(correct_chunks) if correct_chunks else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    mrr = calculate_mrr(retrieved_chunks, correct_chunks)
    return precision, recall, mrr, f1


def evaluate_retrieval_batch(eval_data, ss):
    precisions = []
    recalls = []
    mrrs = []
    f1s = []
    verdicts = []
    all_correct_chunks = []
    all_retrieved_chunks = []
    for i, item in enumerate(tqdm(eval_data, desc="Evaluating Retrieval")):
        correct_chunks = item["correct_chunks"]
        retrieved_chunks = ss.retrieveContent(item["question"])
        precision, recall, mrr, f1 = evaluate_retrieval(retrieved_chunks, correct_chunks)
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        f1s.append(f1)
        verdicts.append(True if f1 >= 0.3 else False)
        all_correct_chunks.append(correct_chunks)
        all_retrieved_chunks.append(retrieved_chunks)
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    avg_f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    return (
        avg_precision,
        avg_recall,
        avg_mrr,
        avg_f1,
        precisions,
        recalls,
        f1s,
        mrrs,
        verdicts,
        all_correct_chunks,
        all_retrieved_chunks,
    )