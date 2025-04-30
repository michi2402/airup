import os
import yaml
from debugpy.common.json import JsonObject
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer, models, util
from util.pubmed_api import PubMedAPI
from util.query_preprocessing import preprocess_query_for_pubmed
from util.word2vec import load_word2vec_model
import logging
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_questions(file_path: str) -> JsonObject:
    """
    Load evaluation questions from a file.
    :param file_path: the path to the file containing questions
    :return: a list of questions
    """
    with open(file_path, 'r') as f:
        questions = json.load(f)
    return questions


def load_model(model_path: str) -> SentenceTransformer:
    """
    Load the pre-trained model.
    :param model_path: the path to the pre-trained model
    :return: a SentenceTransformer model
    """

    word_embedding_model = models.Transformer(model_path)

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def fetch_pubmed_documents(query: str) -> Dictionary:
    """
    Fetch PubMed documents based on a query.
    :param query: the query to search for
    :return: a dictionary of documents of the form {document_id: document_text}
    """

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def infer_top_k_documents(model: SentenceTransformer, query_embedding, documents: dict, k: int = 10) -> list:
    """
    Infer the top k documents based on the query embedding.
    :param model: the SentenceTransformer model
    :param query_embedding: the embedding of the query
    :param documents: a dictionary of documents of the form {document_id: document_text}
    :param k: number of top documents to return
    :return: a list of tuples (document_id, similarity_score)
    """
    similarities = dict()
    for doc_id, json_doc in documents.items():
        doc_text = json_doc.get("abstract", "")
        doc_embedding = model.encode(doc_text, convert_to_tensor=True)
        similarities[doc_id] = util.pytorch_cos_sim(query_embedding, doc_embedding)

    # save 10 best results
    best_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    for doc_id, score in best_results:
        logger.debug(f"Document ID: {doc_id}, Similarity Score: {score.item()}")
    return best_results

def infer_top_k_snippets(model: SentenceTransformer, query_embedding, documents: dict, k: int = 10) -> list:
    # TODO: maybe use sliding window approach to get relevant snippets (dont forget offset adaption)
    """
    Infer the top k snippets based on the query embedding.
    :param model: the SentenceTransformer model
    :param query_embedding: the embedding of the query
    :param documents: a dictionary of documents of the form {document_id: document_text}
    :param k: number of top snippets to return
    :return: a list of tuples (document_id, snippet_text, similarity_score)
    """

    all_snippets = []
    for doc_id, json_doc in documents.items():
        #process title
        doc_title = json_doc.get("title", "")
        title_embedding = model.encode(doc_title, convert_to_tensor=True)
        title_similarity = util.pytorch_cos_sim(query_embedding, title_embedding)
        all_snippets.append((doc_id, doc_title, title_similarity, "title", 0, 0 + len(doc_title)))

        # process abstract
        doc_abstract = json_doc.get("abstract", "")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', doc_abstract)
        sentences = [s for s in sentences if len(s.split()) > 3]
        if not sentences:
            continue

        for sentence in sentences:
            offset_in_begin_section = doc_abstract.find(sentence)
            offset_in_end_section = offset_in_begin_section + len(sentence)
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, sentence_embedding)
            all_snippets.append((doc_id, sentence, similarity, "abstract", offset_in_begin_section, offset_in_end_section))

    top_k_snippets = sorted(all_snippets, key=lambda x: x[2], reverse=True)[:k]
    for doc_id, snippet, score, section, o_b, o_e in top_k_snippets:
        logger.debug(f"Document ID: {doc_id}, Snippet: {snippet}, Similarity Score: {score.item()}, Section: {section}, Offset: {o_b}-{o_e}")
    return top_k_snippets

def save_results_to_json(question_id: str, question: str, document_ids: list[str], snippets: list, file_path: str):
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

    data["questions"].append(new_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    email = os.environ.get("STUDENT_EMAIL")
    if not email:
        raise ValueError("Please set the STUDENT_EMAIL environment variable.")
    pubmed_api = PubMedAPI(email=email, max_results=100)

    logger.info("Loading configuration...")
    config = load_config('config.yaml')

    logger.info("Loading Questions...")
    questions = load_questions(config['questions_path'])

    logger.info("Loading Word2Vec model...")
    w2v_model = load_word2vec_model(config['word2vec_model_path'])

    logger.info("Loading SentenceTransformer model...")
    model = load_model(config['model_name']) #TODO: change to fine-tuned model

    for question in questions.get("questions", []):
        question_id = question.get("id", "")
        question_text = question.get("body", "")

        query = preprocess_query_for_pubmed(question_text, model=w2v_model, max_terms=5)
        documents = pubmed_api.fetch_pubmed(query=query)
        question_embedding = model.encode(question_text, convert_to_tensor=True)
        best_docs = infer_top_k_documents(model, question_embedding, documents, k=10)

        filtered_documents_for_snippets = {
            doc_id: json_doc for doc_id, json_doc in documents.items() if doc_id in [doc[0] for doc in best_docs]
        }
        best_snippets = infer_top_k_snippets(model, question_embedding, filtered_documents_for_snippets, k=10)

        # save json file with results
        save_results_to_json(question_id, question_text, best_docs, best_snippets, config['result_dir'])



if __name__ == "__main__":
    main()