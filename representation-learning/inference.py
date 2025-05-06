import os
import yaml
from debugpy.common.json import JsonObject
from sentence_transformers import SentenceTransformer, models
from util.pubmed_api import PubMedAPI
from util.query_preprocessing import preprocess_query_for_pubmed
from util.word2vec import load_word2vec_model
from util.inference_utils import rank_abstracts, rank_snippet, save_results_to_json_util
import logging
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


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    email = os.environ.get("STUDENT_EMAIL")
    if not email:
        raise ValueError("Please set the STUDENT_EMAIL environment variable.")
    pubmed_api = PubMedAPI(email=email, max_results=100)

    logger.info("Loading configuration...")
    config = load_config('config.yaml')

    #delete results file if present
    if os.path.isfile(config['result_dir']):
        os.remove(config['result_dir'])
        logger.info(f"Deleted existing results file: {config['result_dir']}")

    logger.info("Loading Questions...")
    questions = load_questions(config['train_data_path'])

    logger.info("Loading Word2Vec model...")
    w2v_model = load_word2vec_model(config['word2vec_model_path'], config['w2v_model_output_path'])

    logger.info("Loading SentenceTransformer model...")
    model = load_model(config['output_dir'] + "/my_model")

    for question in questions.get("questions", [])[:10]:
        question_id = question.get("id", "")
        question_text = question.get("body", "")

        query = preprocess_query_for_pubmed(question_text, model=w2v_model, max_terms=5)
        documents = pubmed_api.fetch_pubmed(query=query)
        if not documents or len(documents) == 0:
            logger.warning(f"No documents found for query: {query}")
            continue

        hd = 0
        for doc_id, json_doc in documents.items():
            if question["documents"] and doc_id in question["documents"]:
                hd += 1
        logger.info(f"Question ID: {question_id}, Found {hd} out of {len(question['documents'])} relaevant docs")

        best_docs = rank_abstracts(documents, question_text, model, 10)
        filtered_documents_for_snippets = {
            doc_id: json_doc for doc_id, json_doc in documents.items() if doc_id in [doc[0] for doc in best_docs]
        }
        best_snippets = rank_snippet(filtered_documents_for_snippets, question_text, model, 10)
        save_results_to_json_util(question_id, question_text, best_docs, best_snippets, config['result_dir'], query=query)


if __name__ == "__main__":
    main()
