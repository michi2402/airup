"""
util script add missing questions to the results file
"""

import json

from debugpy.common.json import JsonObject


def load_file(file_path: str) -> JsonObject:
    with open(file_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    return training_data


def main():
    actual_question = load_file("input/BioASQ-task13bPhaseA-testset4.json")

    my_question = load_file("output/results.json")

    for question in actual_question.get("questions", []):
        question_id = question['id']
        question_text = question['body']

        #find question in results data
        actual_result = next((q for q in my_question.get("questions", []) if q.get("id") == question_id), {})
        if not actual_result:
            # append to the end of the document
            my_question["questions"].append({
                "id": question_id,
                "type": question["type"],
                "body": question_text,
                "documents": [],
                "snippets": [],
            })

    with open("output/updated_results.json", 'w') as f:
        json.dump(my_question, f, indent=4)

if __name__ == "__main__":
    main()