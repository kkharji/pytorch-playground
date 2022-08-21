import numpy as np
import torch
import torch.nn as nn
from dataset import ChatDataset

from intents import Intent, get_intents
from model import NeuralNet
from nlp import NLP
from numpy import float32
from numpy._typing import NDArray
from shared import Xy, Label, Word, DEVICE
from torch.utils.data import DataLoader
from torch.optim import Adam


class Trainer:
    epochs_num: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 8
    workers_num: int = 0  # more and training take too long
    hidden_size: int = 8
    words: list[Word] = []
    labels: list[Label] = []
    xy: list[Xy] = []
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    nlp = NLP()

    def dataset(self) -> ChatDataset:
        return ChatDataset(
            nlp=self.nlp,
            words=self.words,
            labels=self.labels,
            xy=self.xy,
        )

    def train_loader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.workers_num,
            shuffle=True,
        )

    def process(self, intents: list[Intent]):
        "Process given training data (intents)"

        blacklist = {k: False for k in ["?", "!", ".", ","]}
        filter_fn = lambda w: blacklist.get(w, True)
        all_words, labels, xy = set(), list(), list()

        for intent in intents:
            label = intent["label"]
            intent_words: list[Word] = []

            labels.append(label)

            for pattern in intent["patterns"]:
                words = self.nlp.tokenize(pattern, filter_fn)
                intent_words.extend(words)

            all_words.update(set(intent_words))
            xy.append(Xy(intent_words, label))

        self.words = sorted(all_words)
        self.labels = sorted(labels)
        self.xy = xy

    def train(self) -> tuple[NeuralNet, ChatDataset]:
        "Train the ML model. NOTE: process need to be called"

        dataset = self.dataset()
        loader = self.train_loader(dataset)
        model = NeuralNet(dataset, self.hidden_size).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        global loss
        # Train
        for epoch in range(self.epochs_num):
            for (words, labels) in loader:
                words = words.to(DEVICE)
                labels = labels.to(dtype=torch.long).to(DEVICE)

                # Forward pass
                outputs = model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs_num}], Loss: {loss.item():.4f}")

            elif epoch == self.epochs_num:
                print(f"final loss: {loss.item():.4f}")

        return model, dataset

    def save(self, model, dataset):
        save_path = "data.pth"
        torch.save(
            {
                "model_state": model.state_dict(),
                "input_size": dataset.input_size,
                "hidden_size": self.hidden_size,
                "output_size": dataset.output_size,
                "words": self.words,
                "labels": self.labels,
            },
            save_path,
        )

        print(f"training complete. file saved to {save_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.process(get_intents())
    model, dataset = trainer.train()
    trainer.save(model, dataset)
