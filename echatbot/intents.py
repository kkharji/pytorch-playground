import json
from typing import TypedDict
from shared import DEFAULT_INTENTS_PATH, Label


class Intent(TypedDict):
    label: Label
    patterns: list[str]
    responses: list[str]


def get_intents(path=DEFAULT_INTENTS_PATH) -> list[Intent]:
    with open(path, "r") as fp:
        return json.load(fp)


def intent_to_raw_data(path=DEFAULT_INTENTS_PATH) -> None:
    data = []
    for intent in get_intents(path):
        for pattern in intent["patterns"]:
            data.append({"text": pattern, "label": intent["label"]})
    with open("intents-raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# intent_to_raw_data()
