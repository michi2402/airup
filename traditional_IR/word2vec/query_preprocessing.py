from gensim.models import KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, RegexpParser

from .word2vec import expand_query_with_word2vec

nltk.download('punkt')
nltk.download('punkt_tab')
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

# extract key terms and expand them
def extract_key_terms_and_expand(question: str, model: KeyedVectors = None, max_terms: int = 10):
    key_terms = extract_key_terms(question, max_terms)
    expanded_key_terms = []

    # iterate throught the terms in the key_terms list and for each term generate a list of expanded terms
 
    expanded_key_terms = [expand_query_with_word2vec(term, model) for term in key_terms]  
            

    return expanded_key_terms


