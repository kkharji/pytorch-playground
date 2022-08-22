from typing import Any, NewType, TypedDict

import torch

# WARN: Either it's not working correctly or metal actually slower
USE_MPS = False

DEFAULT_TRAIN_DATA_RESULT_PATH = "echatbot/resources/data.pth"
DEFAULT_INTENTS_PATH = "echatbot/resources/intents.json"

if torch.backends.mps.is_available() and USE_MPS:
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

Label = NewType("Label", str)
Word = NewType("Word", str)


class TrainingData(TypedDict):
    model_state: dict[str, Any]
    input_size: int
    hidden_size: int
    output_size: int
    words: list[Word]
    labels: list[Label]


def notify(*args):
    print(*args)
    print("-" * 25)
