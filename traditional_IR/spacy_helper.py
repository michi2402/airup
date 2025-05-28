import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from config import spacy_model

nlp = spacy.load(spacy_model)
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True})
nlp.add_pipe("abbreviation_detector")

def apply_lemmatization(text: str) -> str:
    """
    Apply lemmatization to the input text using spaCy.
    :param text: The input text to lemmatize.
    :return: The lemmatized text.
    """
    preprocessed_text = nlp(text)
    return [token.lemma_ for token in preprocessed_text]    

def extract_entitites(text: str) -> list[str]:
    """
    Extract entities from the input text using spaCy.
    :param text: The input text to extract entities from.
    :return: A list of extracted entities.
    """

    doc = nlp(text)
    return [ent.text for ent in doc.ents]