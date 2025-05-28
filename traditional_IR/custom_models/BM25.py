import math

# BASIC BM25 Algorithm
# common ranges: 0.5 < b < 0.8 and 1.2 < k1 < 2.0
# # k1 is a parameter that controls the term frequency saturation effect. It determines how much the term frequency contributes to the score.
# b is a parameter that controls the length normalization. It determines how much the document length affects the score. If b = 0, no length normalization is applied. 
from utils import apply_stemming, apply_lemmatization, split_document_into_sentences
from document import Document

class BM25:
    def __init__(self, preprocessed_documents: list[Document]):

        self.preprocessed_documents = preprocessed_documents

        # each doc has an attribute preprocessed_title and preprocessed_abstract
        # combine these to calculate the length of the document
        self.docs_lengths = [len(doc.preprocessed_title) + len(doc.preprocessed_abstract) for doc in preprocessed_documents]
        self.avg_doc_length = sum(self.docs_lengths) / len(self.docs_lengths)
        self.term_frequencies = self._calculate_term_frequencies()
        self.doc_frequencies = self._calculate_doc_frequencies()
        self.total_documents = len(self.preprocessed_documents)
      

    def _calculate_term_frequencies(self):
        term_frequencies = []
        for doc in self.preprocessed_documents:
            tf = {}
            # Combine title and abstract tokens
            combined_tokens = doc.preprocessed_title + doc.preprocessed_abstract
            for term in combined_tokens:
                tf[term] = tf.get(term, 0) + 1
            term_frequencies.append(tf)
        return term_frequencies

    def _calculate_doc_frequencies(self):
        df = {}
        for doc in self.preprocessed_documents:
            # Combine title and abstract tokens
            combined_tokens = set(doc.preprocessed_title + doc.preprocessed_abstract)
            for term in combined_tokens:
                df[term] = df.get(term, 0) + 1
        return df

    def score(self, preprocessed_query, k1=1.5, b=0.75):
        scores = []
        for i, _ in enumerate(self.preprocessed_documents):
            score = 0
            for term in preprocessed_query:
                if term in self.term_frequencies[i]:
                    tf = self.term_frequencies[i][term]
                    df = self.doc_frequencies.get(term, 0)
                    idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * self.docs_lengths[i] / self.avg_doc_length)
                    score += idf * (numerator / denominator)
            scores.append(score)
        return scores
