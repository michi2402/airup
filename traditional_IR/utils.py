import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from document import Document
from snippet import Snippet
import json
import os
from config import PUBMED_DOC_URL_PREFIX
import requests
from config import API_KEY, E_FETCH_URL, RETMODE, RETMAX, SORT, E_SEARCH_URL
import datetime
from spacy_helper import apply_lemmatization

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


def apply_stemming(words):
    """
    Applies stemming to a list of words using PorterStemmer.

    Args:
        words (list of str): List of words to stem.

    Returns:
        list of str: List of stemmed words.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]



def preprocess_text(text: str, use_stemming=True, use_lemmatization=False):
    # Apply tokenization
    # remove stop words, e.g. the, is, ... that appear very often and have not a high semantic meaning
    # remove question mark
    # stop_words = set(nltk.corpus.stopwords.words('english'))

    # Apply stemming or lemmatization based on the flags
    if use_stemming and use_lemmatization:
        raise ValueError(
            "Both use_stemming and use_lemmatization cannot be True at the same time.")

    if use_stemming:
        tokens = word_tokenize(text.lower())
        return apply_stemming(tokens)
    elif use_lemmatization:
        return apply_lemmatization(text)
    else:
        return tokens


def preprocess_document(document: Document, use_stemming=True, use_lemmatization=False) -> Document:

    preprocessed_title = preprocess_text(
        document.title, use_stemming, use_lemmatization)
    preprocessed_abstract = preprocess_text(
        document.abstract, use_stemming, use_lemmatization)

    # combine title and abstract into a single corpus list[str]
    preprocessed_corpus = preprocessed_title + preprocessed_abstract

    # Add title and abstract together as the corpus
    # Create a new Document object with preprocessed title and abstract
    preprocessed_document = Document(
        id=document.id,
        title=document.title,
        abstract=document.abstract,
        preprocessed_title=preprocessed_title,
        preprocessed_abstract=preprocessed_abstract,
        preprocessed_corpus=preprocessed_corpus
    )

    return preprocessed_document


def make_file(directory: str, filename: str, prefix_path_with_date_time: bool = True, add_stats: bool = False) -> str:
    """
    Create a file path with the given file name.

    Args:
        directory (str): The directory where the file will be created.    
        file_name (str): The name of the file.
        prefix_path_with_date_time (bool): Whether to prefix the file path with the current date and time.

    Returns:
        str: The full path to the created file.
    """
    if prefix_path_with_date_time:
        current_date_time = datetime.datetime.now()
        # prefix the filename the current date and time
        file_name = f"{current_date_time.strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
    # create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    # create the file path
    file_path = os.path.join(directory, file_name)

    if add_stats:
        data = {"stats": [
            {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            ], "questions": []}
    else:
        data = {"questions": []}
        
    # create the file with the data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()

    # return the path to the file
    return file_path


def save_results_to_json(question_id: str, question: str, documents: list[tuple[int, Document]], snippets: list[tuple[int, Snippet]], filePath: str):
    """
    Save the results to a JSON file.

    Args:
        question_id (str): The ID of the question.
        question (str): The question text.
        documents (list of Document): The list of documents to save.
        snippets (list of Snippet): The list of snippets to save.
        output_path (str): The path to save the JSON file.
    """

    with open(filePath, 'r') as f:
        data = json.load(f)

    result = {
        "body": question,
        "id": question_id,
        "documents": [f"{PUBMED_DOC_URL_PREFIX}{score_doc_tuple[1].id}" for score_doc_tuple in documents],
        "snippets": [
            {
                "offsetInBeginSection": snippet.offsetInBeginSection,
                "offsetInEndSection": snippet.offsetInEndSection,
                "text": snippet.text,
                "beginSection": snippet.beginSection,
                "document": f"{PUBMED_DOC_URL_PREFIX}{snippet.document_id}",
                "endSection": snippet.endSection,
            } for _, snippet in snippets],
    }

    data["questions"].append(result)
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()


def generate_snippets_from_document(document: Document) -> list[Snippet]:
    """
    Generate snippets from a document.

    Args:
        document (Document): The document to generate snippets from.
        windows_size (int): The size of the snippet window.
        step (int): The step size for generating snippets.

    Returns:
        list of Snippet: List of generated snippets.
    """

    sentences_tokenized_abstract = nltk.sent_tokenize(document.abstract)
    # make it a string again for the snippet generation

    sentences_abstract = " ".join(sentences_tokenized_abstract)
    snippets = []

    # keep the title as one snippet
    snippets.append(Snippet(document_id=document.id, beginSection="title", offsetInBeginSection=0,
                    endSection="title", offsetInEndSection=len(document.title), text=document.title))

    for i, sentence in enumerate(sentences_tokenized_abstract):
        # get the start and end index of the sentence in the abstract
        start_index = sentences_abstract.index(sentence)
        end_index = start_index + len(sentence)

        # create a snippet from the sentence
        snippets.append(Snippet(document_id=document.id, beginSection="abstract", offsetInBeginSection=start_index,
                        endSection="abstract", offsetInEndSection=end_index, text=sentence))

    return snippets


def retrieve_document_by_id(document_id: str) -> Document:
    """
    Retrieve a document by its ID.

    Args:
        documents (list of Document): The list of documents to search in.
        document_id (str): The ID of the document to retrieve.

    Returns:
        Document: The retrieved document, or None if not found.
    """

    try:
        xml_document_response = requests.get(E_FETCH_URL, params={
            "db": "pubmed", "api_key": API_KEY, "id": document_id, "retmode": RETMODE})
        xml_document_response.raise_for_status()

        article_title: str = xml_document_response.text.split("<ArticleTitle>")[1].split(
            "</ArticleTitle>")[0] if "<ArticleTitle>" in xml_document_response.text else ""
        article_abstract: str = xml_document_response.text.split("<AbstractText>")[1].split(
            "</AbstractText>")[0] if "<AbstractText>" in xml_document_response.text else ""

        # Create a Document object for the retrieved document
        document = Document(
            id=document_id, title=article_title, abstract=article_abstract)

        return document

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving document with ID {document_id}: {e}")
        return None
    except IndexError as e:
        print(f"Error parsing document with ID {document_id}: {e}")
        return None


def retrieve_document_id_list(query: str) -> list[str]:
    xml_response = requests.get(E_SEARCH_URL, params={
                                "db": "pubmed", "api_key": API_KEY, "term": query,  "retmax": RETMAX, "sort": SORT, "retmode": RETMODE})

    # In the XML response, the <IdList> tag contains the list of UIDs that match the query.
    # The UIDs are separated by <Id> tags.
    retrieved_documentId_list = xml_response.text.split("<Id>")
    retrieved_documentId_list = [i.split("</Id>")[0]
                                 for i in retrieved_documentId_list if "</Id>" in i]

    return retrieved_documentId_list


def expand_query(query: str, expanded_key_terms: list[list[str]]) -> str:
    query_with_expanded_terms = word_tokenize(query.lower())

    for i, key_terms in enumerate(expanded_key_terms):
        # check if the key_terms contain the original key term in the query
        for key_term in key_terms:
            if key_term in query_with_expanded_terms:
                # replace the key term with the expanded key terms
                query_with_expanded_terms[query_with_expanded_terms.index(
                    key_term)] = f"{' '.join(key_terms)}"
                break

    # join the query with expanded terms back to a string
    query_with_expanded_terms = ' '.join(query_with_expanded_terms)
    return query_with_expanded_terms


def evaluate_document_retrieval(gold_ids: list[str], best_n_documents: list[tuple[int, Document]]) -> tuple[float, float, float, float]:
    # calculate accuracy of documents in the gold set
    # for each document in the gold set, check if it is in the best_n_documents list
    # if it is, add 1 to the accuracy score
    accuracte_items = 0
    for doc in gold_ids:
        for i, document in best_n_documents:
            if doc == document.id:
                accuracte_items += 1
                break

    accuracy_score = accuracte_items / \
        len(gold_ids) if len(gold_ids) > 0 else 0

    # calculate precision and recall
    # precision = number of relevant documents retrieved / total number of documents retrieved
    precision = accuracte_items / \
        len(best_n_documents) if len(best_n_documents) > 0 else 0
    # recall = number of relevant documents retrieved / total number of relevant documents in the gold set
    recall = accuracte_items / len(gold_ids) if len(gold_ids) > 0 else 0

    # caluclate f1 score
    # f1 score = 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    return accuracy_score, precision, recall, f1_score


def store_stats_to_json(file_path: str, accuracy: float, precision: float, recall: float, f1_score: float):
    """
    Store the stats to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        accuracy_score (float): The accuracy score.
        precision (float): The precision score.
        recall (float): The recall score.
        f1_score (float): The F1 score.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # update the stats with the new values
    data["stats"][0]["accuracy"] = accuracy
    data["stats"][0]["precision"] = precision
    data["stats"][0]["recall"] = recall
    data["stats"][0]["f1"] = f1_score

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()


def max_normalization(best_n_documents_sorted: list[tuple[int, Document]]) -> list[tuple[int, Document]]:
    """
    Normalize the scores of the documents to a range of 0 to 1.
    Take the max score as reference
    Score can be negative, so we need to add the absolute value of the min score to all scores
    """
    max_score = best_n_documents_sorted[0][0]
    # normalize the scores to a range of 0 to 1
    normalized_scores = []
    for score, document in best_n_documents_sorted:
        # normalize the score to a range of 0 to 1
        normalized_score = (score + abs(min(best_n_documents_sorted, key=lambda x: x[0])[0])) / \
            (max_score +
             abs(min(best_n_documents_sorted, key=lambda x: x[0])[0]))
        normalized_scores.append((normalized_score, document))

    return normalized_scores


def apply_drop_off(sorted_items: list[tuple[float, object]], relevance_threshold=0.7, drop_off_ratio=0.2, snippet_min_relevance = 0) -> list[tuple[float, object]]:
    """
    Assumes a sorted list of tuples (score, item) in descending order.
    Implement a top-n scoring drop-off to select the most relevant items.
    If item at position i+1 has a score that is less than the score of item i * drop_off_ratio, 
    or falls below the relevance threshold, then drop every item after i+1.
    For snippets, we implement a snippet_min_relevance threshold to filter out snippets that are considered not relevant enough.
    """
    if len(sorted_items) == 0:
        return []

    best_score = sorted_items[0][0]
    relevance_score_based_off_best = best_score * relevance_threshold

    
    if len(sorted_items) == 1:
        if snippet_min_relevance > 0:
            if best_score < snippet_min_relevance:
                return []
        return sorted_items


    # start with item at index 1 to compare to item at index 0
    for index, tuple_entry in enumerate(sorted_items[1:], start=1):
        score = tuple_entry[0]
        current_item = tuple_entry[1]
        prev_item = sorted_items[index-1][1]
        prev_score = sorted_items[index-1][0]

        drop_off = (prev_score - score) / prev_score
    
        if snippet_min_relevance > 0:
            if best_score < snippet_min_relevance:
                return []

        if score < relevance_score_based_off_best:
            # drop all items after this one
            return sorted_items[:index]

        # check if the score of the current item is under score of the previous item considering the drop_off_ratio
        if drop_off > drop_off_ratio:
            # drop all items after this one
            return sorted_items[:index]

    return sorted_items
