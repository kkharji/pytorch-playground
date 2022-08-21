from typing import NewType

import torch

Label = NewType("Label", str)
Word = NewType("Word", str)

# Either it's not working correctly or metal actually slower
USE_MPS = False

if torch.backends.mps.is_available() and USE_MPS:
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class Xy:
    def __init__(self, words: list[Word], label: Label) -> None:
        self.x = words
        self.y = label
