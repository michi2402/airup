import json
from utils import preprocess_document, preprocess_text, save_results_to_json, generate_snippets_from_document, retrieve_document_by_id, expand_query, retrieve_document_id_list, evaluate_document_retrieval, make_file, store_stats_to_json, max_normalization, apply_drop_off
from sklearn.model_selection import train_test_split
from document import Document
from word2vec.query_preprocessing import preprocess_query_for_pubmed, extract_key_terms_and_expand
from word2vec.word2vec import load_word2vec_model
from config import WORD2VEC_MODEL_PATH, WORD2VEC_MODEL_SAVE_PATH, USE_LEMMATIZATION, USE_STEMMING, OUTPUT_DIRECTORY_PATH, OUTPUT_FILENAME, DROP_OFF_RATIO, RELEVANCE_THRESHOLD, SNIPPET_MIN_SCORE
import os
from rank_bm25 import BM25Plus as BM25Plus
from snippet import Snippet

n = 10  # number of documents/snippets to return
question_index_incl = 60
question_index_excl = 60
# create file for storing the results

results_file_path = make_file(
    'output', OUTPUT_FILENAME, prefix_path_with_date_time=True, add_stats=False)

# load the word2vec model to preprocess the query and enrich it
word2vec_model = load_word2vec_model(
    WORD2VEC_MODEL_PATH, WORD2VEC_MODEL_SAVE_PATH)


# Load the JSON file containing the training data
with open('./BioASQ-training13b/testset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

    questions = data['questions']
    curr_questions = questions[question_index_incl:]

for question in curr_questions:
    # Query api and retrieve n documents for the question from the api
    query = question['body']
    print("Query:", query)

    question_id = question['id']

    # use word2vec to find relevant documents on pubmed based on the preprocessed query
    preprocessed_query_for_pubmed = preprocess_query_for_pubmed(
        query, model=word2vec_model, max_terms=10)

    # retrieve the document ids from the api using the preprocessed query
    retrieved_documentId_list = retrieve_document_id_list(
        preprocessed_query_for_pubmed)

    if retrieved_documentId_list != []:

        preprocessed_documents: Document = []
        # corpus holds the preprocessed title + preprocessed abstract for each document
        corpus_docs = []

        # Iterate over all the UIDs with an index in the list and retrieve the document content for each UID.
        for i, uid in enumerate(retrieved_documentId_list):
            print("Processing document number", i)
            document = retrieve_document_by_id(uid)

            # preprocess the document: use tokenization, remove stop words, lowercase, apply stemming or lemmatization
            preprocessed_document = preprocess_document(
                document, use_stemming=USE_STEMMING, use_lemmatization=USE_LEMMATIZATION)

            if preprocessed_document is not None:
                preprocessed_documents.append(preprocessed_document)
                corpus_docs.append(preprocessed_document.preprocessed_corpus)

        # extract key terms from the query and expand them with word2vec
        expanded_key_terms: list[list[str]] = extract_key_terms_and_expand(
            query, model=word2vec_model, max_terms=10)

        # change the original query by substituting the key terms in the query with the expanded key terms
        # this should help us find relevant documents with BM25 for expanded key terms
        # tokenize the query and convert it to lowercase
        query_with_expanded_terms = expand_query(query, expanded_key_terms)

        # preprocess the query with the same preprocessing steps as the documents
        preprocessed_query = preprocess_text(
            query, use_stemming=USE_STEMMING, use_lemmatization=USE_LEMMATIZATION)

        # Step 1 - Document level retrieval
        # Rank whole documents based on their relevance to the query.
        # need to return <= 10 documents with the highest scores
        bm25_docs = BM25Plus(corpus_docs, k1=1.5, b=0.75, delta=1)
        scores_docs = bm25_docs.get_scores(preprocessed_query)

        # list of tuples (score, document) for the top n documents with the highest scores
        best_n_documents = []
        # iterate through the scores and get the top n documents with the highest scores
        for i, score in enumerate(scores_docs):
            best_n_documents.append((score, preprocessed_documents[i]))
        best_n_documents = sorted(
            best_n_documents, key=lambda x: x[0], reverse=True)[:n]

        # normalize the scores to be between 0 and 1 where 1 is the max score
        # best_n_documents_normalized_scores = max_normalization(best_n_documents)
        best_n_documents_normalized_scores = best_n_documents

        # apply the drop_off function
        best_n_documents = apply_drop_off(
            best_n_documents_normalized_scores, drop_off_ratio=DROP_OFF_RATIO, relevance_threshold=RELEVANCE_THRESHOLD)

        # Step 2 - Snippet level retrieval
        # need to return <= 10 snippets of the documents with the highest scores
        # reuse existing BM25 Setup and treat snippets as "mini-documents"
        # run the query against these mini documents
        # snippets are sentences or even parts of sentences from the documents

        overall_best_snippets = []

        # best_n_documents is a list of tuples (score, document)
        for _, tuple_entry in enumerate(best_n_documents):
            current_doc = tuple_entry[1]
            snippets: list[Snippet] = generate_snippets_from_document(
                current_doc)

            #  add the snippets to the corpus for the bm25 model
            corpus_snippets = []
            for snippet in snippets:
                preprocessed_snippet = preprocess_text(
                    snippet.text, use_stemming=USE_STEMMING, use_lemmatization=USE_LEMMATIZATION)
                corpus_snippets.append(preprocessed_snippet)

            bm25_mini_docs = BM25Plus(corpus_snippets, k1=1.5, b=0.75, delta=1)

            # run the query against the snippets
            scores_snippets = bm25_mini_docs.get_scores(preprocessed_query)

            # get the top k snippets from the documents with the highest scores
            best_n_snippets_for_doc = []
            for i, score in enumerate(scores_snippets):
                best_n_snippets_for_doc.append((score, snippets[i]))
            best_n_snippets_for_doc = sorted(
                best_n_snippets_for_doc, key=lambda x: x[0], reverse=True)[:n]

            best_n_snippets_for_doc = apply_drop_off(
                best_n_snippets_for_doc, drop_off_ratio=DROP_OFF_RATIO, relevance_threshold=RELEVANCE_THRESHOLD, snippet_min_relevance=SNIPPET_MIN_SCORE)

            overall_best_snippets.extend(best_n_snippets_for_doc)

        # get the top n snippets from the overall best snippets
        overall_best_snippets = sorted(
            overall_best_snippets, key=lambda x: x[0], reverse=True)[:n]

        overall_best_snippets = apply_drop_off(
            overall_best_snippets, drop_off_ratio=DROP_OFF_RATIO, relevance_threshold=RELEVANCE_THRESHOLD)

        save_results_to_json(question_id, query, best_n_documents,
                             snippets=overall_best_snippets, filePath=results_file_path)
    else:
        save_results_to_json(
            question_id, query, [], snippets=[], filePath=results_file_path)