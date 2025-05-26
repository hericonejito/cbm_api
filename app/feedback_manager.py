import os
import json
import uuid

def save_feedback(feedback_data, folder='feedback'):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{uuid.uuid4()}.json")
    with open(file_path, "w") as f:
        json.dump(feedback_data, f)

def load_all_feedback(folder='feedback'):
    feedback = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                feedback.append(json.load(f))
    return feedback
