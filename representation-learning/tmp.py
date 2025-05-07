"""
old approach stuff
"""

import json
import os

import numpy as np
import yaml
from debugpy.common.json import JsonObject
from sentence_transformers import SentenceTransformer, models
from util.query_preprocessing import preprocess_query_for_pubmed
from util.word2vec import load_word2vec_model
from util.pubmed_api import PubMedAPI
import logging
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    email = os.environ.get("STUDENT_EMAIL")
    if not email:
        raise ValueError("Please set the STUDENT_EMAIL environment variable.")
    pubmed_api = PubMedAPI(email)

    logger.info("Loading configuration...")
    config = load_config('config.yaml')

    logger.info("Loading Questions...")
    questions = load_questions(config['questions_path'])

    logger.info("Loading Word2Vec model...")
    w2v_model = load_word2vec_model(config['word2vec_model_path'], config['w2v_model_output_path'])

    logger.info("Loading Model...")
    word_embedding_model = models.Transformer(config['model_name'])
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    # fetch articles for each question
    for question in questions.get("questions", []):
        query_id = question.get("id")
        question_text = question.get("body")
        query = preprocess_query_for_pubmed(question_text, w2v_model, 5)
        docs = pubmed_api.fetch_pubmed(query)

        corpus_texts = []
        pmids = []
        for doc_id, json_obj in docs.items():
            corpus_texts.append(f"{json_obj['title']}. {json_obj['abstract']}")
            pmids.append(doc_id)

        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=False, show_progress_bar=True)

        # buiold faiss  index
        corpus_embeddings = np.array(corpus_embeddings).astype("float32")
        dimension = corpus_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(corpus_embeddings)

        question_embedding = model.encode([question], convert_to_tensor=False).astype("float32")

        k = 10
        # docs
        distance, indices = index.search(question_embedding, k)

        logger.info(f"Top {k} articles for question '{question}':")
        for i in range(k):
            idx = indices[0][i]
            pmid = pmids[idx]
            title, abstract = corpus_texts[idx].split('. ', 1)
            logger.info(f"PMID: {pmid}, Title: {title}, Abstract: {abstract}")

        # snippets
        question_vector = question_embedding[0]
        top_snippets = []
        for doc_id, json_obj in docs.items():
            sentences = sent_tokenize(json_obj["abstract"])
            for sent in sentences:
                sent_vec = model.encode([sent], convert_to_tensor=False)
                score = cosine_similarity([question_vector], [sent_vec[0]])[0][0]
                top_snippets.append((score, sent, doc_id))

        top_snippets.sort(reverse=True)
        top_snippets = top_snippets[:10]
        for score, snippet, doc_id in top_snippets:
            logger.info(f"ID: {doc_id}, Snippet: {snippet}, Score: {score}")

if __name__ == "__main__":
    main()