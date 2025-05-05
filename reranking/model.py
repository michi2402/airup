from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from reranking.config import DATA_PATH, RERANKER_PATH
from reranking.data_preprocessing import get_testing_data

model_dir = RERANKER_PATH
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()  # set to evaluation mode ??

def rerank(question, snippets, docs):
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
    reranked_docs = sorted(zip(snippets, docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return reranked_docs

test_data = get_testing_data(DATA_PATH)

for question in test_data:
    doc_scores = {}

    snippets =  [x[0] for x in test_data[question]]
    docs =  [x[1] for x in test_data[question]]
    labelled_snippets = test_data[question]

    # Get model predictions
    scores = rerank(question, snippets, docs)
    scores.sort(key=lambda x: x[2], reverse=True)
    labelled_snippets.sort(key=lambda x: x[2], reverse=True)


    print(f"Question: {question}")
    for i, (snippet, doc, score) in enumerate(scores, 1):
        print(f"{i}. Score: {score} - {snippet} ({doc})")
    print("-" * 140)
    for i, (snippet, doc, score) in enumerate(labelled_snippets, 1):
        print(f"{i}. Score: {score} - {snippet} ({doc})")

    print("#" * 140)


