import mlx.core
import mlx.nn
import torch
import numpy as np

from pathlib import Path
import mlx.core as mx
from model import Bert, load_model
from datasets import DatasetDict, Dataset
import mlx
# from mlx.nn.losses import cross_entropy
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, filename="train.log", filemode="w")


class ModelForTokenClassification(nn.Module):
    def __init__(self, num_labels, max_length=512) -> None:
        super().__init__()
        self.bert_model, self.bert_tokenizer = load_model(
            "bert-base-uncased",
            "weights/bert-base-uncased.npz")
        self.bert_model.freeze()
        self.bert_model_output_dim = 768
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, num_labels)

    def __call__(self, tokens) -> mx.array:
        # tokens = self.bert_tokenizer(
        #     words, return_tensors="np", padding=True, is_split_into_words=True, truncation=True)

        # tokens = {key: mx.array(v) for key, v in tokens.items()}

        output, _ = self.bert_model(**tokens)
        output = self.dropout(output)
        output = self.linear1(output)
        logits = mx.softmax(output, axis=-1)

        return logits


def compute_loss(logits: mx.array, y: mx.array) -> float:
    logits_flatten = logits.flatten()
    y_flatten = y.flatten()

    index_to_check = [idx for idx, element in enumerate(
        y_flatten) if element == -100]
    return mlx.nn.losses.cross_entropy(logits_flatten[index_to_check], y_flatten[index_to_check]).mean()


def loss_function(model, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return compute_loss(logits, y)


if __name__ == "__main__":
    print("**** Training ****")

    train_dataset = Dataset.load_from_disk("./hf_train_ner_dataset")

    test_dataset = Dataset.load_from_disk("./hf_test_ner_dataset")
    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=21)

    loss_and_grad_fn = nn.value_and_grad(model, loss_function)

    optimizer = mlx.optimizers.Adam(learning_rate=5e-5)
    batch_size = 32
    num_epochs = 1
    epoch_iterator = tqdm(iterable=range(num_epochs),
                          total=num_epochs, desc="Epochs")
    for epoch in epoch_iterator:
        epoch_loss = 0
        train_batch_iterator = tqdm(range(0, len(train_dataset), batch_size), total=len(
            train_dataset)//batch_size, desc="Batches")
        model.train()
        model.bert_model.freeze()
        for i in train_batch_iterator:
            # st = time.time()
            slice = train_dataset[i: i + batch_size]

            if "words" in slice.keys():
                slice.pop("words")
            if "ner_labels" in slice.keys():
                slice.pop("ner_labels")

            labels = slice.pop("labels")
            for key in slice.keys():
                slice[key] = mx.array([mx.array(array)
                                      for array in slice[key]])

            labels = mx.array([mx.array(array) for array in labels])
            loss_value, grads = loss_and_grad_fn(model, slice, labels)
            optimizer.update(model, grads)
            epoch_loss += loss_value
            # et = time.time()
            # train_batch_iterator.set_postfix(loss=epoch_loss.tolist())
            break

        model.eval()
        validation_loss = 0
        for i in range(0, len(test_dataset), batch_size):
            test_slice = test_dataset[i: i + batch_size]
            test_slice.pop("words")
            test_slice.pop("ner_labels")

            test_labels = test_slice.pop("labels")
            test_labels = mx.array([mx.array(array) for array in test_labels])
            for key in test_slice.keys():
                test_slice[key] = mx.array(
                    [mx.array(array) for array in test_slice[key]])

            output = model(test_slice)
            test_loss = compute_loss(output, test_labels)
            validation_loss += test_loss
            break
        # epoch_iterator.set_postfix(loss=epoch_loss.tolist())
        logging.info(f"\tEpoch: {epoch} | train_loss: {
                     epoch_loss.tolist()} | val_loss: {validation_loss.tolist()}")
        break
