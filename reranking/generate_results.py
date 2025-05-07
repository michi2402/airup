import json
import os

from alive_progress import alive_bar

from reranking.config import *
from reranking.data_preprocessing import process_data, process_data_for_docs
from reranking.model import load_model, rerank, rerank_docs


def generate_results():
    data = process_data(INPUT_FILE_PATH)
    data_docs = process_data_for_docs(INPUT_FILE_DOCS)
    tokenizer, model = load_model(RERANKER_PATH)

    with alive_bar(len(data), force_tty=True) as bar:
        for question in data:
            snippets = [x[0] for x in data[question]]
            docs = [x[1] for x in data[question]]
            q_ids = [x[3] for x in data[question]]
            snippet_data = [x[4] for x in data[question]]

            scores = rerank(tokenizer, model, question, snippets, docs, q_ids, snippet_data)
            doc_scores = {}
            snippet_scores = []
            for _, doc, score, _, snippet_data in scores:
                doc_scores[doc] = doc_scores.get(doc, 0) + score
                snippet_data["score"] = score
                snippet_scores.append(snippet_data)


            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            sorted_snippets = sorted(snippet_scores, key=lambda x: x["score"], reverse=True)[:10]
            save_results_to_json_util(scores[0][3],
              question,
              list(map(lambda x: x[0], sorted_docs)),
              sorted_snippets,
              OUTPUT_FILE_PATH
            )
            bar()

    with alive_bar(len(data_docs), force_tty=True) as bar:
        for question in data_docs:
            doc_links = [x[0] for x in data_docs[question]]
            docs = [x[1] for x in data_docs[question]]
            q_ids = [x[3] for x in data_docs[question]]

            scores = rerank_docs(tokenizer, model, question, doc_links, docs, q_ids)
            doc_scores = {}
            snippet_scores = []
            for doc, doc_link, score, q_id in scores:
                doc_scores[doc_link] = score

            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[0], reverse=True)[:10]
            save_results_to_json_util(scores[0][3],
              question,
              list(map(lambda x: x[0], sorted_docs)),
        [],
              "./output/results_docs.json"
            )
            bar()


def save_results_to_json_util(question_id: str, question: str, document_ids: list[str], snippets, file_path: str):
    """
    Save the results to a JSON file.
    :param results: the results to save
    :param output_path: the path to save the JSON file
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {"questions": []}

    new_entry = {
        "body": question,
        "id": question_id,
        "documents": document_ids,
        "snippets": snippets,
    }

    data["questions"].append(new_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def evaluate_output(result_file_path: str):
    result_data = {"questions": []}
    if os.path.isfile(result_file_path):
        with open(result_file_path, 'r') as f:
            result_data = json.load(f)["questions"]

    with (open(DATA_PATH, 'r') as f):
        data = json.load(f)["questions"]

        total_docs = 0
        total_snippets = 0

        for question in result_data:
            qid = question["id"]
            qdata = next((x for x in list(data) if x["id"] == qid), None)

            docs_gold = set(json.dumps(obj, sort_keys=True) for obj in qdata["documents"])
            docs_reranked = set(json.dumps(obj, sort_keys=True) for obj in question["documents"][:10])
            correct_docs = len(docs_gold & docs_reranked)
            total_docs += correct_docs

            snippets_gold = set(json.dumps(obj, sort_keys=True) for obj in map(lambda x: x["text"], qdata["snippets"]))
            snippets_reranked = set(json.dumps(obj, sort_keys=True) for obj in list(map(lambda x: x["text"], question["snippets"]))[:10])
            correct_snippets = len(snippets_gold & snippets_reranked)
            total_snippets += correct_snippets

            print(f"{qid}: {correct_docs}/10 docs, {correct_snippets}/10 snippets")

        print(f"TOTAL: {total_docs} docs, {total_snippets} snippets")


#generate_results()
evaluate_output(INPUT_FILE_PATH)
print("*" * 120)
evaluate_output(OUTPUT_FILE_PATH)
print("*" * 120)
evaluate_output("./output/results_docs.json")