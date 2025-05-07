import json
import matplotlib.pyplot as plt

from reranking.config import DATA_PATH
from reranking.data_preprocessing import get_testing_data
from reranking.model import load_model, rerank


def plot_loss():
    with open("./reranker/checkpoint-3519/trainer_state.json") as f:
        state = json.load(f)

    log_history = state["log_history"]

    # Extract data
    steps = []
    train_loss = []
    eval_loss = []

    for entry in log_history:
        if "loss" in entry:
            steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])

    # Plot
    plt.plot(steps, train_loss, label="Train Loss")
    if eval_loss:
        plt.plot(steps[:len(eval_loss)], eval_loss, label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


def eyeball_evaluation_reranker():
    test_data = get_testing_data(DATA_PATH)
    tokenizer, model = load_model()

    for question in test_data:
        snippets = [x[0] for x in test_data[question]]
        docs = [x[1] for x in test_data[question]]
        labelled_snippets = test_data[question]

        # Get model predictions
        scores = rerank(tokenizer, model, question, snippets, docs)
        scores.sort(key=lambda x: x[2], reverse=True)
        labelled_snippets.sort(key=lambda x: x[2], reverse=True)

        print(f"Question: {question}")
        for i, (snippet, doc, score) in enumerate(scores, 1):
            print(f"{i}. Score: {score} - {snippet} ({doc})")
        print("-" * 140)
        for i, (snippet, doc, score) in enumerate(labelled_snippets, 1):
            print(f"{i}. Score: {score} - {snippet} ({doc})")

        print("#" * 140)
