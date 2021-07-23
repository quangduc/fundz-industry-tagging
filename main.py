import json
import os
import pdb
from thinc.schedules import compounding
import spacy
from spacy.training import Example
import random

FOLDER_FILE_PATH = "/Users/leducquang/Downloads/training"
TRAINING_DATA = []

index = 0
labels = {}
for subdir, dirs, files in os.walk(FOLDER_FILE_PATH):
    for file in files:
        if index >= 10000:
            break

        with open(os.path.join(subdir, file)) as f:
            items = json.load(f)
            for item in items:
                industries = {group.lower().replace(" ", "-"): True for group in item['industry_groups']}
                labels.update(industries)
                TRAINING_DATA.append([item["description"], industries])
                index += 1

from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
config = {
   "threshold": 0.5,
   "model": DEFAULT_MULTI_TEXTCAT_MODEL,
}
nlp = spacy.blank("en")
category = nlp.add_pipe("textcat_multilabel", config=config, last=True)
for label in labels:
    category.add_label(label)

nlp.begin_training()

for itn in range(100):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
    batch_sizes = compounding(4.0, 32.0, 1.001)

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=batch_sizes):
        texts = [nlp.make_doc(text) for text, entities in batch]
        annotations = [{"cats": entities} for text, entities in batch]

        # uses an example object rather than text/annotation tuple
        examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
            texts, annotations
        )]
        nlp.update(examples, losses=losses)
    if itn % 20 == 0:
        print(losses)

nlp.to_disk(".")
pdb.set_trace()
