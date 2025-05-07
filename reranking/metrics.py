from tqdm import tqdm
from alive_progress import alive_bar
from reranking.config import DATA_PATH, CROSS_ENCODER_PRETRAINED_MODEL_MINILM_MARCO, RERANKER_PATH
from reranking.data_preprocessing import get_testing_data
from reranking.model import load_model, rerank


def compute_metrics(model_path):
    test_data = get_testing_data(DATA_PATH)
    tokenizer, model = load_model(model_path)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    num_questions = len(test_data)
    with alive_bar(num_questions, force_tty=True) as bar:
        for i, question in enumerate(test_data):
            snippets = [x[0] for x in test_data[question]]
            docs = [x[1] for x in test_data[question]]
            true_relevance = test_data[question]

            scores = rerank(tokenizer, model, question, snippets, docs)

            #collect tp and fp
            for j, tr in enumerate(true_relevance):
                _, _, tr_score = tr
                _, _, pr_score = scores[j]
                if tr_score > 0 and pr_score > 0:
                    tp += 1
                if tr_score <= 0 < pr_score:
                    fp += 1
                if tr_score < 0 and pr_score < 0:
                    tn += 1
                if tr_score > 0 >= pr_score:
                    fn += 1
            bar()

        precision_total = tp / (tp + fp)
        accuracy_total = (tp + tn) / (tp + tn + fp + fn)
        recall_total = tp/ (tp + fn)
        f1_total = 2 * (precision_total * recall_total) / (precision_total + recall_total)


    metrics = {
        "Accuracy": accuracy_total,
        "Precision": precision_total,
        "Recall": recall_total,
        "F1": f1_total
    }
    return metrics

metrics_marco = compute_metrics(CROSS_ENCODER_PRETRAINED_MODEL_MINILM_MARCO)
metrics_cool = compute_metrics(RERANKER_PATH)
print(f"{CROSS_ENCODER_PRETRAINED_MODEL_MINILM_MARCO}: {metrics_marco}")
print(f"RERANKER: {metrics_cool}")
