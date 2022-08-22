import nlp
import torch
import torch.nn as nn

from dataset import ChatDataset
from intents import Intent, get_intents
from model import NeuralNet
from shared import TrainingData, Label, Word, DEVICE, notify
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import long


class Trainer:
    epochs_num: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 8
    workers_num: int = 0  # more and training take too long
    hidden_size: int = 8
    words: list[Word] = []
    labels: list[Label] = []
    xy: list[tuple[list[Word], Label]] = []
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    def process(self, intents: list[Intent]):
        "Process given training data (intents)"

        for intent in intents:
            label = intent["label"]
            self.labels.append(label)

            for pattern in intent["patterns"]:
                words = nlp.tokenize(pattern)

                self.xy.append((words, label))
                self.words.extend([nlp.stem(w) for w in words])

        self.words = sorted(set(self.words))
        self.labels = sorted(set(self.labels))
        notify(len(self.xy), "patterns")
        notify(len(self.labels), "tags:", self.labels)
        notify(len(self.words), "unique stemmed words:", self.words)

    def dataset(self) -> ChatDataset:
        return ChatDataset(
            words_store=self.words,
            labels=self.labels,
            xy=self.xy,
        )

    def train_loader(self, dataset: ChatDataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.workers_num,
            shuffle=True,
        )

    def model(self, dataset) -> NeuralNet:
        return NeuralNet(
            input_size=dataset.input_size,
            hidden_size=self.hidden_size,
            output_size=dataset.output_size,
        )

    def training_data(self, dataset: ChatDataset, model: NeuralNet) -> TrainingData:
        "the model state, input,hidden, and output sizes, as well as x and y data."

        return {
            "model_state": model.state_dict(),
            "input_size": dataset.input_size,
            "hidden_size": self.hidden_size,
            "output_size": dataset.output_size,
            "words": self.words,
            "labels": self.labels,
        }

    def train(self) -> tuple[NeuralNet, ChatDataset]:
        "Train the ML model. NOTE: process need to be called"

        dataset = self.dataset()
        loader = self.train_loader(dataset)
        model = self.model(dataset)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        global loss

        for epoch in range(self.epochs_num):
            for (words, labels) in loader:
                # Fix datatype and set device
                words, labels = words.to(DEVICE), labels.to(dtype=long).to(DEVICE)
                # Forward pass
                outputs = model(words)
                loss = self.criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                if epoch + 1 == self.epochs_num:
                    print(f"Final Loss: {loss.item():.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{self.epochs_num}]: Loss: {loss.item():.4f}")

        return model, dataset

    def save(self, model: NeuralNet, dataset: ChatDataset) -> None:
        "Serialize and save training resulting data"

        save_path = "data.pth"
        torch.save(self.training_data(dataset, model), save_path)
        print(f"training complete. file saved to {save_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.process(get_intents())
    model, dataset = trainer.train()
    trainer.save(model, dataset)
