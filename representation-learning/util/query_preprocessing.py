import spacy
from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, RegexpParser

from .word2vec import expand_query_with_word2vec
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

nlp = spacy.load("en_ner_bc5cdr_md")

def preprocess_query_for_pubmed(question: str, model: KeyedVectors = None, max_terms: int = 10) -> str:
    """
    Preprocess a question into a PubMed query using key term extraction and word2vec expansion.
    :param question: the question to preprocess
    :param model: word2vec model for term expansion
    :param max_terms: maximum number of key terms to extract
    :return: the processed query string
    """
    question = question.replace("?", "")
    _, question = question.split(" ", 1) #remove "ask" word (e.g. "what", "who", "where", "when", "how", "why", ...)
    question = question.lower()

    key_terms = extract_key_terms_spacy(question, max_terms)
    logger.info(f"Extracted key terms: {key_terms}")
    if len(key_terms) <= 1:
        key_terms.extend(extract_key_terms(question, max_terms))
        logger.info(f"Extracted key terms: {key_terms}")

    query = ""
    for term in key_terms:
        expanded_terms = expand_query_with_word2vec(term, model)[:3]
        query += f"({' OR '.join(expanded_terms)}) AND "
        #query += f"{term} AND "

    # Remove the trailing " AND "
    if query.endswith(" AND "):
        query = query[:-5]
    return query

def extract_key_terms_new(question: str, max_terms: int = 5) -> list[str]:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(question.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens#[:max_terms]


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

    grammar = "NP: {<JJ>*<NN.*>+}"
    chunker = RegexpParser(grammar)
    tree = chunker.parse(pos_tags)

    noun_phrases = [' '.join(leaf[0] for leaf in subtree.leaves())
                    for subtree in tree.subtrees()
                    if subtree.label() == 'NP']

    # Filter stopword-only or very short phrases
    noun_phrases = [
        phrase for phrase in noun_phrases
        if not all(word in stop_words for word in phrase.split())
           and len(phrase) > 2
    ]

    # Deduplicate and sort by length
    unique_phrases = sorted(set(noun_phrases), key=lambda x: len(x), reverse=True)[:max_terms]

    return unique_phrases

def extract_key_terms_spacy(question: str, max_terms: int = 5) -> list[str]:
    """
    Extract key terms from a question using spaCy's dependency parsing.
    :param question: the question to process
    :param max_terms: maximum number of key terms to extract
    :return: a list of key terms
    """

    doc = nlp(question)
    noun_phrases = [ent.text for ent in doc.ents]
    #unique_phrases = sorted(set(noun_phrases), key=lambda x: len(x), reverse=True)

    return noun_phrases
