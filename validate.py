import openai as OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

json_file = "qa_validation.json"  # Path to the JSON file with correct QA pairs

def load_correct_answers(json_file):
    """
    Loads the correct questions and answers from a JSON file.

    Parameters:
    - json_file (str): Path to the JSON file

    Returns:
    - dict: Dictionary with the question as key and the correct answer as value
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question"]: item["answer"] for item in data}


def validate_answer(question, given_answer):
    """
    Validates a given answer against the correct answer from a JSON file.

    Parameters:
    - question (str): The question being answered
    - given_answer (str): The provided answer

    Returns:
    - dict: Dictionary with "score" (1-10) and "feedback" (string)
    """
    # Load correct answers
    correct_answers = load_correct_answers(json_file)

    if question not in correct_answers:
        return {
            "score": 0,
            "feedback": "This question is not found in the JSON and cannot be validated."
        }

    correct_answer = correct_answers[question]

    # OpenAI prompt
    prompt = f"""
    You are a strict validator for an AI system. Evaluate the following question and the given answer
    against the correct answer.

    **Correct Question and Answer:**
    Question: {question}
    Correct Answer: {correct_answer}

    **Given Answer:**
    {given_answer}

    **Evaluate the given answer:**
    - Is the answer factually correct? (Yes/No)
    - Are there any inaccuracies or misinterpretations?
    - Does it contain unnecessary information that is not in the correct answer?
    - Provide a score from 1 to 10 for accuracy and relevance (10 = perfect, 1 = completely incorrect)
    - Give a short feedback on any mistakes and how the answer can be improved.

    Format the output as a JSON object with fields: "score" (1-10) and "feedback" (string).
    """

    response = OpenAI.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON output from OpenAI response
    # result = json.loads(response.choices[0].message.content)

    print(response.choices[0].message.content)  # Returns a dictionary with "score" and "feedback"
