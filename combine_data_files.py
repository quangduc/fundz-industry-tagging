import json
import os
import pdb
from thinc.schedules import compounding
import spacy
from spacy.training import Example
import random

FOLDER_FILE_PATH = "/Users/leducquang/Downloads/training_copy"
TRAINING_DATA = []

index = 0
labels = {}
items = []
for subdir, dirs, files in os.walk(FOLDER_FILE_PATH):
    for file in files:
        if index >= 10000:
            break

        with open(os.path.join(subdir, file)) as f:
            items.extend(json.load(f))

f = open("/Users/leducquang/Downloads/training_copy/training_data.json", "a")
f.write(json.dumps(items))
f.close()
