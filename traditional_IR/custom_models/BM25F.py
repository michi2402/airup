import math
from collections import defaultdict

class BM25F:
    def __init__(self, documents, field_weights=None, k1=1.5, b=0.75):
        """
        Initialize BM25F with documents and parameters.
        
        :param documents: List of dictionaries where keys are field names and values are field texts.
        :param field_weights: Dictionary of field weights (e.g., {'title': 2.0, 'abstract': 1.0}).
        :param k1: Term frequency saturation parameter.
        :param b: Length normalization parameter.
        """
        self.documents = documents
        self.field_weights = field_weights or {}
        self.k1 = k1
        self.b = b
        self.doc_lengths = defaultdict(lambda: defaultdict(float))
        self.avg_doc_lengths = defaultdict(float)
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.doc_count = len(documents)
        self.idf = defaultdict(float)
        
        self._initialize()

    def _initialize(self):
        """
        Precompute document lengths, average lengths, and inverted index.
        """
        field_totals = defaultdict(float)
        
        for doc_id, doc in enumerate(self.documents):
            for field, text in doc.items():
                terms = text.split()
                self.doc_lengths[doc_id][field] = len(terms)
                field_totals[field] += len(terms)
                
                for term in terms:
                    self.inverted_index[term][field].append(doc_id)
        
        for field, total_length in field_totals.items():
            self.avg_doc_lengths[field] = total_length / self.doc_count
        
        self._compute_idf()

    def _compute_idf(self):
        """
        Compute inverse document frequency (IDF) for each term.
        """
        for term, field_docs in self.inverted_index.items():
            doc_freq = sum(len(doc_ids) for doc_ids in field_docs.values())
            self.idf[term] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def score(self, query, doc_id):
        """
        Compute BM25F score for a document given a query.
        
        :param query: List of query terms.
        :param doc_id: Document ID to score.
        :return: BM25F score.
        """
        score = 0.0
        for term in query:
            if term not in self.inverted_index:
                continue
            
            term_score = 0.0
            for field, weight in self.field_weights.items():
                tf = self._term_frequency(term, doc_id, field)
                doc_len = self.doc_lengths[doc_id][field]
                avg_len = self.avg_doc_lengths[field]
                norm_tf = tf / (1 + self.k1 * ((1 - self.b) + self.b * (doc_len / avg_len)))
                term_score += weight * norm_tf
            
            score += self.idf[term] * term_score
        return score

    def _term_frequency(self, term, doc_id, field):
        """
        Compute term frequency for a term in a specific document field.
        
        :param term: Term to compute frequency for.
        :param doc_id: Document ID.
        :param field: Field name.
        :return: Term frequency.
        """
        return self.documents[doc_id][field].split().count(term)

    def rank(self, query):
        """
        Rank all documents for a given query.
        
        :param query: List of query terms.
        :return: List of (doc_id, score) tuples sorted by score in descending order.
        """
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in range(self.doc_count)]
        return sorted(scores, key=lambda x: x[1], reverse=True)