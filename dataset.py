import numpy as np
from torch.utils.data import Dataset


class ChatDataset(Dataset):
    def __init__(self, nlp, words, labels, xy):
        x_data, y_data = [], []

        for item in xy:
            # X: bag of words for each pattern_sentence
            bag = nlp.bag(item.x, words)
            x_data.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = labels.index(item.y)
            y_data.append(label)

        self.x_data, self.y_data = np.array(x_data), np.array(y_data)
        self.n_samples: int = len(x_data)

        self.input_size = len(x_data[0])
        self.output_size = len(labels)

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
