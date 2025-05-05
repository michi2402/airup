import json
import os.path
import random
from collections import defaultdict
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split

from reranking.config import *
from reranking.question_clustering import get_similar_questions

number_neg_snippets = 5

def get_all_questions(filename):
    with open(filename) as f:
        data = json.load(f)

    questions = []
    questions_with_snippets = []
    for i, q in enumerate(data["questions"]):
        questions.append(q["body"])
        snippets = q.get("snippets", [])
        questions_with_snippets.append(snippets)

    return questions, questions_with_snippets

def save_data_from_file(filename):
    with open(filename) as f:
        data = json.load(f)

    rows = []
    question_groups = defaultdict(list)

    #positive samples
    for q in data["questions"]:
        question = q["body"]
        snippets = q.get("snippets", [])
        for snippet in snippets:
            question_groups[question].append({
                "question": question,
                "snippet": snippet["text"],
                "label": float(1),
                "document": snippet["document"],
            })

    #negative samples
    all_questions, questions_with_snippets = get_all_questions(filename)
    similar_question_groups = get_similar_questions(all_questions)
    for i, similar_qs in enumerate(similar_question_groups):
        question = all_questions[i]
        for j, q_index in enumerate(similar_qs):
            random_snippet = random.choice(questions_with_snippets[q_index])
            label = LABEL_CLOSE_NEG if j < AMOUNT_SAME_SIMILAR else LABEL_MID_NEG if j < AMOUNT_MID_SIMILAR else LABEL_FAR_NEG
            question_groups[question].append({
                "question": question,
                "snippet": random_snippet["text"],
                "label": float(label),
                "document": random_snippet["document"],
            })

    train_questions, test_questions  = train_test_split(all_questions, test_size=0.1)

    train_dataset = Dataset.from_list([row for q in train_questions for row in question_groups[q]])
    test_dataset = Dataset.from_list([row for q in test_questions for row in question_groups[q]])

    filename_for_save = filename[len(DATA_FOLDER):].split(".")[0]
    train_dataset.save_to_disk(filename_for_save + TRAIN_FILE_EXTENSION)
    test_dataset.save_to_disk(filename_for_save + TEST_FILE_EXTENSION)
    return test_dataset, all_questions

def get_training_data(filename):
    filename_for_save = filename[len(DATA_FOLDER):].split(".")[0]
    if os.path.exists(filename_for_save +  TRAIN_FILE_EXTENSION):
        train_dataset  = load_from_disk(filename_for_save + TRAIN_FILE_EXTENSION)
    else:
        train_dataset, _ = save_data_from_file(filename)
    return train_dataset

def get_testing_data(filename):
    filename_for_save = filename[len(DATA_FOLDER):].split(".")[0]
    if os.path.exists(filename_for_save + TEST_FILE_EXTENSION):
        test_dataset = load_from_disk(filename_for_save + TEST_FILE_EXTENSION)
        grouped_data = defaultdict(list)

        for example in test_dataset:
            question = example["question"]
            snippet = example["snippet"]
            label = example["label"]
            document = example["document"]
            grouped_data[question].append((snippet, document, label))
        return grouped_data



