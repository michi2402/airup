from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from reranking.data_preprocessing import get_testing_data

model_dir = "./reranker"  # or wherever you saved it
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()  # set to evaluation mode



def rerank(question, docs):
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
    reranked_docs = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return reranked_docs

test_data = get_testing_data("training13b.json")

for question in test_data:
    passages =  [x[0] for x in test_data[question]]
    labelled_passages = test_data[question]

    # Get model predictions
    scores = rerank(question, passages)
    scores.sort(key=lambda x: x[1], reverse=True)
    labelled_passages.sort(key=lambda x: x[1], reverse=True)


    print(f"Question: {question}")
    for i, (passage, score) in enumerate(scores, 1):
        print(f"{i}. Score: {score} - {passage}")
    print("-" * 140)
    for i, (passage, score) in enumerate(labelled_passages, 1):
        print(f"{i}. Score: {score} - {passage}")

    print("#" * 140)


