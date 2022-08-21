# import pytest
from nlp import NLP


nlp = NLP()


def test_tokenize():
    input = "How long does shipping take?"
    output = nlp.tokenize(input, lambda _: True)
    assert output == ["How", "long", "does", "shipping", "take", "?"]


def test_stem():
    input = ["organize", "organizes", "organizing"]
    output = [nlp.stem(word) for word in input]
    assert output == ["organ", "organ", "organ"]


def test_bag():
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    expected = [0, 1, 0, 1, 0, 0, 0]
    output = nlp.bag(sentence, words)
    assert expected == list(output)


def test_stem_with_different_letter():
    input = ["organize", "organizes", "organising"]
    output = [nlp.stem(w) for w in input]
    assert output == ["organ", "organ", "organis"]
