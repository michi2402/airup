import json
import os.path
import random
from collections import defaultdict
from datasets import Dataset, load_from_disk

number_neg_snippets = 5
label_for_neg = 0

def save_data_from_file(filename):
    with open(filename) as f:
        data = json.load(f)

    rows = []

    for q in data["questions"]:
        question = q["body"]
        snippets = q.get("snippets", [])
        for i, snippet in enumerate(snippets):
            rows.append({
                "question": question,
                "passage": snippet["text"],
                "label": float(1)
            })



        neg_snippets = []
        for i in range(number_neg_snippets):
            random_question = random.choice(list(data["questions"]))
            random_snippet = random.choice(list(random_question["snippets"]))
            neg_snippets.append(random_snippet)

        for i, neg_snippet in enumerate(neg_snippets):
            rows.append({
                "question": question,
                "passage": neg_snippet["text"],
                "label": float(label_for_neg)
            })


    df= Dataset.from_list(rows)

    split = df.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]

    filename_for_save = filename[len("../BioASQ-training13b/"):].split(".")[0]
    train_dataset.save_to_disk(filename_for_save + "_train_dataset")
    test_dataset.save_to_disk(filename_for_save + "_test_dataset")
    return test_dataset

def get_training_data(filename):
    filename_for_save = filename[len("../BioASQ-training13b/"):].split(".")[0]
    if os.path.exists(filename_for_save +  "_train_dataset"):
        train_dataset = load_from_disk(filename_for_save + "_train_dataset")
    else:
        train_dataset = save_data_from_file(filename)
    return train_dataset

def get_testing_data(filename):
    filename_for_save = filename[len("../BioASQ-training13b/"):].split(".")[0]
    if os.path.exists(filename_for_save + "_test_dataset"):
        test_dataset = load_from_disk(filename_for_save + "_test_dataset")#%%
        grouped_data = defaultdict(list)

        for example in test_dataset:
            # Normalize question text
            question = example["question"]
            passage = example["passage"]
            label = example["label"]
            grouped_data[question].append((passage, label))
        return grouped_data



