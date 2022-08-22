import numpy as np
import nlp
from torch.utils.data import Dataset

from shared import Label, Word, notify


class ChatDataset(Dataset):
    def __init__(self, words_store, labels, xy: list[tuple[list[Word], Label]]):
        x_data, y_data = [], []

        for (words, label) in xy:
            # X: bag of words for each pattern_sentence
            bag = nlp.bag(words, words_store)
            x_data.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = labels.index(label)
            y_data.append(label)

        self.x_data, self.y_data = np.array(x_data), np.array(y_data)
        self.n_samples: int = len(x_data)

        self.input_size = len(x_data[0])
        self.output_size = len(labels)

        notify(f"input_size: {self.input_size}, output_size: {self.output_size}")

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
