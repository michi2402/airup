import json
import logging
import os

from sympy.core.random import random
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator(Dataset):
    def __init__(self, data_path, output_path, generate_new_dataset=False):
        self.generate_new_dataset = generate_new_dataset
        if not generate_new_dataset and os.path.exists(output_path):
            logger.info("Loading dataset from file...")
            with open(output_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Loaded dataset with {len(self.data)} samples")
            return

        logger.info("Generating dataset...")
        self.data_path = data_path
        self.data = self.generate_dataset(data_path)
        self.write_dataset_to_file(output_path)
        logger.info(f"Loaded dataset with {len(self.data)} samples")

    def generate_dataset(self, data_path):

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []

        for question_obj in data['questions']:
            processed_data.extend(self.generate_positive_examples(question_obj))
            processed_data.extend(self.generate_negative_examples(question_obj, data, 5))

        return processed_data

    def generate_positive_examples(self, question_obj):
        """
        Generates a positive examples for the given question from the training data.
        :param question_obj: The question object from the dataset.
        :return: A list containing objects with question_id, question, document ID, document snippet, and label.
        """
        question = question_obj['body']
        question_id = question_obj['id']

        positive_examples = []
        snippets_obj = question_obj['snippets']
        for snippet in snippets_obj:
            document_id = snippet['document']
            document_snippet = snippet['text']
            positive_examples.append({
                'question_id': question_id,
                'question': question,
                'document_id': document_id,
                'document_snippet': document_snippet,
                'label': 1
            })
        return positive_examples

    def generate_negative_examples(self, question_obj, data, max_negative_examples_count=5):
        """
        Generates negative examples for the given question from the training data.

        This is done by taking snippets from other questions meaning they are not relevant to the current question.

        :param question_obj: The question object from the dataset.
        :param data: The entire dataset.
        :param max_negative_examples_count: The maximum number of negative examples to generate.
        :return: A list containing objects with question_id, question, document ID, document snippet, and label.
        """

        question = question_obj['body']
        question_id = question_obj['id']

        negative_examples = []
        negative_examples_count = 0
        for q_obj in data['questions']:
            q_id = q_obj['id']
            if q_id == question_id:
                continue

            if negative_examples_count >= max_negative_examples_count:
                break

            if random() < 0.9:
                continue

            for snippet in q_obj['snippets']:
                if negative_examples_count >= max_negative_examples_count:
                    break

                if random() < 0.8:
                    continue
                document_id = snippet['document']
                document_snippet = snippet['text']
                negative_examples.append({
                    'question_id': question_id,
                    'question': question,
                    'document_id': document_id,
                    'document_snippet': document_snippet,
                    'label': 0
                })
                negative_examples_count += 1

        if negative_examples_count != max_negative_examples_count:
            logger.warning(f"Generated {negative_examples_count} negative examples instead of {max_negative_examples_count}")
        return negative_examples



    def write_dataset_to_file(self, output_path):
        """
        Write the generated dataset to a file.
        :param output_path: The path where the dataset should be saved.
        """

        # create file if it does not exist
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)