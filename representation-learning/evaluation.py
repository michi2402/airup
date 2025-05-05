import json

import yaml
from debugpy.common.json import JsonObject


def load_file(file_path: str) -> JsonObject:
    with open(file_path, 'r') as f:
        training_data = json.load(f)
    return training_data

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config('config.yaml')
    training_data = load_file(config['train_data_path'])
    results_data = load_file(config['result_dir'])

    doc_hits = 0
    doc_sum = 0
    snippet_hits = 0
    snippet_sum = 0
    per_q_metric = dict()
    for question in training_data.get("questions", []):
        # Extract the question text and expected results
        question_id = question['id']
        question_text = question['body']
        optimal_docs = question['documents']
        optimal_snippets = question['snippets']

        #find question in results data
        actual_result = next((q for q in results_data.get("questions", []) if q.get("id") == question_id), {})
        if not actual_result:
            print(f"Question ID {question_id} not found in results data.")
            continue

        actual_docs = actual_result.get("documents", [])
        actual_snippets = actual_result.get("snippets", [])

        if not actual_docs and not actual_snippets:
            print(f"No documents or snippets found for question ID {question_id}.")
            continue

        d_s = 0
        d_h = 0
        for doc in optimal_docs:
            d_s += 1
            if doc in actual_docs:
                d_h += 1

        s_s = 0
        s_h = 0
        for snippet in optimal_snippets:
            s_s += 1
            if snippet in actual_snippets:
                s_h += 1

        # calculate accuracy for docs and snippets
        doc_accuracy = d_h / d_s if d_s > 0 else 0
        snippet_accuracy = s_h / s_s if s_s > 0 else 0

        per_q_metric[question_id] = {
            "doc_accuracy": doc_accuracy,
            "snippet_accuracy": snippet_accuracy
        }
        doc_hits += d_h
        doc_sum += d_s
        snippet_hits += s_h
        snippet_sum += s_s

    # save stats about model performance to a file
    stats = {
        "doc_accuracy": doc_hits / doc_sum if doc_sum > 0 else 0,
        "snippet_accuracy": snippet_hits / snippet_sum if snippet_sum > 0 else 0,
        "per_question_metrics": per_q_metric
    }
    with open(config['stats_dir'], 'w') as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()