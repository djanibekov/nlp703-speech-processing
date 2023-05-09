# Adapted from https://github.com/iqbalfarz/intent-classification
from pathlib import Path

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

import evaluate

# audio_dataset = load_dataset("akhmedsakip/music-berkeley-emotions")
audio_dataset = load_dataset("akhmedsakip/ravdess-singing-emotions")

label2id, id2label = dict(), dict()
labels = audio_dataset["train"].features["label"].names
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# model_checkpoint = "akhmedsakip/wav2vec2-base-berkeley"
model_checkpoint = "akhmedsakip/wav2vec2-base-ravdess"
# model_checkpoint = "facebook/wav2vec2-base"
# model_checkpoint = "facebook/hubert-large-ls960-ft"


# loading Feature Extractor
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
feature_extractor

# Creating preprocessing function
max_audio_len = 10
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate*int(max_audio_len)),
        # truncation=True, # Uncomment it, If you want to truncate longer audios to max_length
        # padding=True, # Uncomment it, if you want to pad shorter audio to max_length
    )
    return inputs

encoded_audio_dataset = audio_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
encoded_audio_dataset

from collections import Counter

train_dataset = encoded_audio_dataset['train']
test_dataset = encoded_audio_dataset['test']

train_labels = train_dataset['label']
test_labels = test_dataset['label']

train_label_freq = Counter(train_labels)
test_label_freq = Counter(test_labels)

metric = "f1"

from transformers import AutoModelForAudioClassification
from transformers import TrainingArguments
from transformers import Trainer

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

model_name = model_checkpoint.split("/")[-1]
batch_size = 2
args = TrainingArguments(
    f"{model_name}-berkeley-output-{metric}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit = 2,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=100,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model=f"eval_{metric}",
    weight_decay=0.01,
    logging_dir="./logs_berkeley",
    push_to_hub=False,
)

def compute_metrics(eval_pred):
    """
    this method compute metrics and return the result

    In case of accuracy
    """
    metric_type=metric
    metric_loaded = evaluate.load(metric_type)
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    average = "micro"
    
    if metric_type!="accuracy":
        result = metric_loaded.compute(predictions=predictions, references=references, average=average)
    else:
        result = metric_loaded.compute(predictions=predictions, references=references)

    return result
    
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_audio_dataset["train"],
    eval_dataset=encoded_audio_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# trainer.push_to_hub()