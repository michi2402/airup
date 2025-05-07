import json
import os
import logging

import torch
from sentence_transformers import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rank_abstracts(documents, question, model, top_k=10):
    """
    Rank the documents based on their abstracts using cosine similarity.
    :param documents: the documents to rank
    :param question: the question to rank the documents against
    :param model: the model to use for encoding
    :param top_k: the number of top documents to return
    :return: a sorted list of the top_k documents based on their similarity to the question
    """

    question_embedding = model.encode(question)

    doc_items = list(documents.items())
    abstracts = [a.get("abstract", "") for _, a in doc_items]
    abstract_embeddings = model.encode(abstracts)
    sim = util.pytorch_cos_sim(question_embedding, abstract_embeddings)

    sorted_scores, sorted_indices = torch.sort(sim, descending=True)
    sorted_indices = sorted_indices.flatten()

    sorted_docs = [doc_items[i] for i in sorted_indices[:top_k]]
    return sorted_docs

def rank_snippet(documents, question, model, top_k=10):
    """
    Rank the documents based on their snippets using dot score.
    :param documents: the documents to rank
    :param question: the question to rank the documents against
    :param model: the model to use for encoding
    :param top_k: the number of top snippets to return
    :return: a list of the top_k snippets from the documents
    """

    result = []
    all_snippets = []

    for doc_id, json_obj in documents.items():
        abstract = json_obj.get("abstract", "")
        snippets = abstract.split(". ")

        question_embedding = model.encode(question, normalize_embeddings=True)
        snippet_embeddings = model.encode(snippets, normalize_embeddings=True)
        dot_scores = util.dot_score(question_embedding, snippet_embeddings)[0]

        for i, snippet in enumerate(snippets):
            all_snippets.append({
                "document": doc_id,
                "text": snippet,
                "score": dot_scores[i].item(),
                "offsetInBeginSection": abstract.find(snippet),
                "offsetInEndSection": abstract.find(snippet) + len(snippet),
                "beginSection": "abstract",
                "endSection": "abstract",
            })

        title = json_obj.get("title", "")
        title_embedding = model.encode(title, normalize_embeddings=True)
        dot_score = util.dot_score(question_embedding, title_embedding)[0]
        all_snippets.append({
            "document": doc_id,
            "text": title,
            "score": dot_score.item(),
            "offsetInBeginSection": 0,
            "offsetInEndSection": len(title),
            "beginSection": "title",
            "endSection": "title",
        })

    # Sort all snippets by score in descending order and take the top k
    top_snippets = sorted(all_snippets, key=lambda x: x["score"], reverse=True)[:top_k]

    for snippet in top_snippets:
        result.append({
            "document": snippet["document"],
            "text": snippet["text"],
            "offsetInBeginSection": snippet["offsetInBeginSection"],
            "offsetInEndSection": snippet["offsetInEndSection"],
            "beginSection": snippet["beginSection"],
            "endSection": snippet["endSection"],
        })

    return result

def save_results_to_json_util(question_id: str, question: str, question_type: str, document_ids: list[str], snippets: list, file_path: str, query: str = None):
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
        "id": question_id,
        "type": question_type,
        "body": question,
        "documents": [doc_id for doc_id, _ in document_ids],
        #"extra_wuensche": [{doc_id: json_obj.get("abstract", "")} for doc_id, json_obj in document_ids],
        "snippets": snippets,
    }

    if query is not None:
        new_entry["query"] = query

    data["questions"].append(new_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)