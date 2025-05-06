import os
from sentence_transformers import SentenceTransformer, util
import logging
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_top_k_documents(model: SentenceTransformer, question_embedding, synonyms_with_score: dict, documents: dict,
                          k: int = 10) -> list:
    """
    Infer the top k documents based on the query embedding and synonyms.
    :param model: the SentenceTransformer model
    :param question_embedding: the embedding of the query
    :param synonyms_with_score: a dictionary of synonyms with their scores
    :param documents: a dictionary of documents
    :param k: number of top documents to return
    :return: the best k documents
    """
    similarities = dict()
    for doc_id, json_doc in documents.items():
        doc_text = json_doc.get("abstract", "").lower()

        # Replace synonyms in the document text
        logger.debug(f"Original document text: {doc_text}")
        for term, list_of_synonyms in synonyms_with_score.items():
            for synonym, score in list_of_synonyms:
                if synonym in doc_text:
                    doc_text = doc_text.replace(synonym, term)
        logger.debug(f"Modified document text: {doc_text}")
        # Encode the modified document text
        doc_embedding = model.encode(doc_text, convert_to_tensor=True)
        similarities[doc_id] = util.pytorch_cos_sim(question_embedding, doc_embedding)

    # save 10 best results
    best_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    for doc_id, score in best_results:
        logger.debug(f"Document ID: {doc_id}, Similarity Score: {score.item()}")
    return best_results


def infer_top_k_snippets(model: SentenceTransformer, question_embedding, synonyms_with_score: dict, documents: dict,
                         k: int = 10) -> list:
    # TODO: maybe use sliding window approach to get relevant snippets (dont forget offset adaption)
    """
    Infer the top k snippets based on the query embedding.
    :param model: the SentenceTransformer model
    :param question_embedding: the embedding of the query
    :param synonyms_with_score: a dictionary of synonyms with their scores
    :param documents: a dictionary of documents of the form {document_id: document_text}
    :param k: number of top snippets to return
    :return: a list of tuples (document_id, snippet_text, similarity_score)
    """

    all_snippets = []
    for doc_id, json_doc in documents.items():
        # process title
        doc_title = json_doc.get("title", "").lower()
        logger.debug(f"Title: {doc_title}")
        for term, list_of_synonyms in synonyms_with_score.items():
            for synonym, score in list_of_synonyms:
                if synonym in doc_title:
                    doc_title = doc_title.replace(synonym, term)
        logger.debug(f"Modified title: {doc_title}")
        title_embedding = model.encode(doc_title, convert_to_tensor=True)
        title_similarity = util.pytorch_cos_sim(question_embedding, title_embedding)
        all_snippets.append((doc_id, json_doc.get("title", ""), title_similarity, "title", 0, 0 + len(doc_title)))

        # process abstract
        doc_abstract = json_doc.get("abstract", "")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', doc_abstract)
        sentences = [s for s in sentences if len(s.split()) > 3]
        if not sentences:
            continue

        for sentence in sentences:
            offset_in_begin_section = doc_abstract.find(sentence)
            offset_in_end_section = offset_in_begin_section + len(sentence)
            syn_sentence = sentence.lower()
            for term, list_of_synonyms in synonyms_with_score.items():
                for synonym, score in list_of_synonyms:
                    if synonym in syn_sentence:
                        syn_sentence = sentence.replace(synonym, term)
            sentence_embedding = model.encode(syn_sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(question_embedding, sentence_embedding)
            all_snippets.append(
                (doc_id, sentence, similarity, "abstract", offset_in_begin_section, offset_in_end_section))

    top_k_snippets = sorted(all_snippets, key=lambda x: x[2], reverse=True)[:k]
    for doc_id, snippet, score, section, o_b, o_e in top_k_snippets:
        logger.debug(
            f"Document ID: {doc_id}, Snippet: {snippet}, Similarity Score: {score.item()}, Section: {section}, Offset: {o_b}-{o_e}")
    return top_k_snippets


def save_results_to_json(question_id: str, question: str, document_ids: list[str], snippets: list, file_path: str, query: str = None):
    """
    Save the results to a JSON file in the format specified by bioasq.
    :param question_id: the ID of the question
    :param question: the question text
    :param document_ids: the top k document IDs (tuple (document_id, similarity_score))
    :param snippets: the top k snippets (tuple (document_id, snippet_text, similarity_score))
    :param file_path: the path to save the JSON file to
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
        "documents": [doc_id for doc_id, score in document_ids],
        "snippets": [{
            "document": doc_id,
            "text": snippet,
            "offsetInBeginSection": offsetInBeginSection,
            "offsetInEndSection": offsetInEndSection,
            "beginSection": section,
            "endSection": section,
        } for doc_id, snippet, score, section, offsetInBeginSection, offsetInEndSection in snippets]
    }

    if query is not None:
        new_entry["query"] = query

    data["questions"].append(new_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

