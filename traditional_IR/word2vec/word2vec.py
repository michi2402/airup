from gensim.models import KeyedVectors
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_word2vec_model(model_path: str, save_path: str) -> KeyedVectors:
    """
    Load a Word2Vec model from the specified path.
    :param model_path: Path to the Word2Vec model directory.
    :return: KeyedVectors object containing the word vectors.
    """

    words = []
    vectors = []

    # Check if the model is already saved
    try:
        kv = KeyedVectors.load(save_path, mmap='r')
        logger.info(f"Loaded Word2Vec model from {save_path}")
        return kv
    except FileNotFoundError:
        logger.info(f"Word2Vec model not found at {save_path}, loading from {model_path}")

    with open(model_path + '/types.txt', 'r', encoding='utf-8') as f_types, open(model_path + '/vectors.txt', 'r', encoding='utf-8') as f_vectors:
        for word_line, vector_line in zip(f_types, f_vectors):
            word = word_line.strip()
            vector = np.fromstring(vector_line.strip(), sep=' ')
            words.append(word)
            vectors.append(vector)

    vectors = np.vstack(vectors)
    assert len(words) == vectors.shape[0], f"Mismatch: {len(words)} words vs {vectors.shape[0]} vectors"
    kv = KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(words, vectors)
    logger.info(f"Loaded {len(words)} words and vectors from {model_path}")
    kv.save(save_path)
    return kv

def expand_query_with_word2vec(term: str, model: KeyedVectors, top_n: int = 5) -> list[str]:
    """
    Expand a query term using a Word2Vec model.
    :param term: The term to expand.
    :param model: The Word2Vec model to use for expansion.
    :param top_n: Number of similar words to retrieve.
    :return: A list of expanded terms.
    """

    #expanded_terms = [term]
    #try:
    #    similar_words = model.most_similar(term, topn=top_n)
    #    for word, score in similar_words:
    #        if score > .6 and len(word) > 2 and word not in expanded_terms:
    #            expanded_terms.append(word)
    #except KeyError:
    #    logger.warning(f"Term '{term}' not found in Word2Vec model vocabulary.")
    #return expanded_terms
    expansion_terms = get_synonyms_with_score(term, model, top_n)
    expanded_terms = [term]
    try:
        for word, score in expansion_terms[term]:
                expanded_terms.append(word)
    except KeyError:
        logger.warning(f"Term '{term}' not found in Word2Vec model vocabulary.")
    return expanded_terms

def get_synonyms_with_score(term: str, model: KeyedVectors, top_n: int = 5) -> dict:
    """
    Get synonyms for a term along with their similarity scores.
    :param term: The term to find synonyms for.
    :param model: The Word2Vec model to use for finding synonyms.
    :param top_n: Number of synonyms to retrieve.
    :return: A dictionary of synonyms and their similarity scores.
    """

    synonyms = dict()
    try:
        similar_words = model.most_similar(term, topn=top_n)
        for word, score in similar_words:
            if score > .6 and len(word) > 2:
                if term in synonyms:
                    synonyms[term].append((word, score))
                else:
                    synonyms[term] = [(word, score)]
    except KeyError:
        logger.warning(f"Term '{term}' not found in Word2Vec model vocabulary.")
    return synonyms