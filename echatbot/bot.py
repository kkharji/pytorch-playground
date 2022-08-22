import random
import signal

import torch
from torch import Tensor
from torch.types import Number
from shared import DEFAULT_TRAIN_DATA_RESULT_PATH

from intents import Intent
from model import NeuralNet

import nlp
from shared import DEVICE, Label, TrainingData


class Bot:
    name = "Daniel"

    def __init__(
        self,
        intents: list[Intent],
        training_objects_filepath=DEFAULT_TRAIN_DATA_RESULT_PATH,
    ) -> None:
        self.intents = intents
        self.data: TrainingData = torch.load(training_objects_filepath)
        self.trained_words = self.data["words"]
        self.trained_labels = self.data["labels"]
        self.model = NeuralNet(
            input_size=self.data["input_size"],
            hidden_size=self.data["hidden_size"],
            output_size=self.data["output_size"],
        )
        self.model.load_state_dict(self.data["model_state"])
        self.model.eval()

    def reply(self, msg: str) -> None:
        print(f"{self.name}: {msg}")

    def should_quit(self, sentence: str):
        return sentence == "quit"

    def predict_label(self, sentence) -> tuple[Label, Number]:
        # Tokenize the sentence
        tokens = nlp.tokenize(sentence)
        # Bag of words
        bag = nlp.bag(tokens, self.trained_words)
        # Redshape the bag
        bag = bag.reshape(1, bag.shape[0])
        # Convert to tensor
        bag = torch.from_numpy(bag).to(DEVICE)
        # Get output
        output: Tensor = self.model(bag)
        # Get the maximum value of all elements in the tensor.
        max = torch.max(output, dim=1)
        # Get predicted index (second memeber of max)
        pred_index = max[1].item()
        # Calcluate probability
        prob = torch.softmax(output, dim=1).to(DEVICE)[0][pred_index].item()
        # Get Label
        label = self.trained_labels[pred_index]

        return label, prob

    def get_response(self, label):
        # Get label using pred_index
        for intent in self.intents:
            if label == intent["label"]:
                return random.choice(intent["responses"])

    def start(self):
        print(f"(type 'quit' at any time to exit)")

        def handler(signum, frame):
            print("")
            self.reply("bye!")
            exit(0)

        signal.signal(signal.SIGINT, handler)

        while True:
            sentence = input("Me: ")
            # Break the loop when it's a quit message
            if self.should_quit(sentence):
                self.reply("bye!")
                break

            # Get predict label and it's probability
            label, prob = self.predict_label(sentence)
            print(prob)

            # Skip probability less then 0.75
            if prob < 0.65:
                self.reply(f"Sorry, I'm not trained to reply to '{sentence}'")
                continue

            response = self.get_response(label)
            if response:
                self.reply(response)
            else:
                self.reply(f"I undersatnd, but I have nothing to say to you")
