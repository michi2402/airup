from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from reranking.config import DATA_PATH, RERANKER_PATH
from reranking.data_preprocessing import get_testing_data

def load_model(model_dir = RERANKER_PATH):
    """Loads a pretrained tokenizer and sequence classification model from a directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def rerank(tokenizer, model, question, snippets, docs, q_ids, snippet_data):
    """Tokenize question-snippet pairs, get relevance scores from model, and return snippets sorted by score"""
    inputs = tokenizer(
        text=[question] * len(snippets),
        text_pair=snippets,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    #rerank by score
    reranked_snippets = sorted(zip(snippets, docs, scores.tolist(), q_ids, snippet_data), key=lambda x: x[1], reverse=True)
    return reranked_snippets

def rerank_docs(tokenizer, model, question, doc_links,docs , q_ids):
    """Tokenize question-doc pairs, get relevance scores from model, and return docs sorted by score"""
    inputs = tokenizer(
        text=[question] * len(docs),
        text_pair=docs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    #rerank by score
    reranked_docs = sorted(zip(docs, doc_links, scores.tolist(), q_ids ), key=lambda x: x[1], reverse=True)
    return reranked_docs


