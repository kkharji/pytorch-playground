import numpy as np
from typing import cast
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from numpy._typing import NDArray
from numpy import float32
from shared import Word

# Need to be called at least once
# nltk.download('punkt')


class NLP:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def tokenize(self, pattern: str) -> list[Word]:
        words = word_tokenize(pattern)
        return cast(list[Word], words)

    def stem(self, word: str) -> list[Word]:
        return self.stemmer.stem(word=word, to_lowercase=True)

    def bag_of_words(self, tokenized_sentence, all_words) -> NDArray[float32]:
        """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
        # Initialize bag with 0 for each word
        bag = np.zeros(len(all_words), dtype=float32)

        # Stem each word
        words = [self.stem(w) for w in tokenized_sentence]

        for idx, word in enumerate(all_words):
            if word in words:
                bag[idx] = 1.0

        return bag
