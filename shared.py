from typing import NewType

import torch

Label = NewType("Label", str)
Word = NewType("Word", str)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class Xy:
    def __init__(self, words: list[Word], label: Label) -> None:
        self.x = words
        self.y = label
