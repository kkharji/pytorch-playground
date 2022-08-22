import sys

from bot import Bot
from intents import get_intents
from train import Trainer

cmd = sys.argv[1]

if cmd == "train":
    intents = get_intents()
    trainer = Trainer()
    trainer.process(intents)
    model, dataset = trainer.train()
    trainer.save(model, dataset)
    sys.exit(0)

elif cmd == "run":
    intents = get_intents()
    Bot(intents).start()

else:
    print("")
    print("Available Commands")
    print("  train: train echatboot neural net")
    print("  run: Run the echatboot")
    print("")
    sys.exit(1)
