from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from reranking.config import DATA_PATH, RERANKER_PATH
from reranking.data_preprocessing import get_testing_data

def load_model(model_dir = RERANKER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # set to evaluation mode ??
    return tokenizer, model

def rerank(tokenizer, model, question, snippets, docs, q_ids, snippet_data):
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
    reranked_docs = sorted(zip(snippets, docs, scores.tolist(), q_ids, snippet_data), key=lambda x: x[1], reverse=True)
    return reranked_docs

def rerank_docs(tokenizer, model, question, doc_links,docs , q_ids):
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


