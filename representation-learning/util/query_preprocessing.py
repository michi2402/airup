from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

from .word2vec import expand_query_with_word2vec

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')


def preprocess_query_for_pubmed(question: str, model: KeyedVectors = None, max_terms: int = 10) -> str:
    """
    Preprocess a question into a PubMed query using key term extraction and word2vec expansion.
    :param question: the question to preprocess
    :param model: word2vec model for term expansion
    :param max_terms: maximum number of key terms to extract
    :return: the processed query string
    """

    key_terms = extract_key_terms(question, max_terms)
    query = ""
    for term in key_terms:
        expanded_terms = expand_query_with_word2vec(term, model)
        query += f"({' OR '.join(expanded_terms)}) AND "

    # Remove the trailing " AND "
    if query.endswith(" AND "):
        query = query[:-5]
    return query

def extract_key_terms(question: str, max_terms: int = 5) -> list[str]:
    """
    Extract key terms from a question using POS tagging and filtering.
    :param question: the question to process
    :param max_terms: maximum number of key terms to extract
    :return: a list of key terms
    """

    clean_q = re.sub(r'[^\w\s]', ' ', question.lower())
    tokens = word_tokenize(clean_q)
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))

    # Only keep nouns (NN, NNS, NNP, NNPS) and adjectives (JJ)
    candidate_terms = [
        word for word, pos in pos_tags
        if (pos.startswith('NN') or pos == 'JJ')
        and word not in stop_words
        and len(word) > 2
    ]

    # Sort by word length (longer = usually more specific biomedical terms)
    sorted_terms = sorted(candidate_terms, key=len, reverse=True)[:max_terms]

    return sorted_terms