import mlx.core
import numpy as np

import mlx.core as mx
from mlx.core import array

from model import load_model
from datasets import Dataset
import mlx
from mlx.optimizers import Adam
from tqdm import tqdm

import logging
import mlx.nn as nn
from mlx.utils import tree_flatten
from typing import Dict, Union


logging.basicConfig(level=logging.INFO, filename="train.log", filemode="w")


class ModelForTokenClassification(nn.Module):
    def __init__(self, num_labels, max_length=512) -> None:
        super().__init__()
        self.bert_model, self.bert_tokenizer = load_model(
            "bert-base-uncased", "weights/bert-base-uncased.npz"
        )
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


def compute_loss(logits: mx.array, y: mx.array) -> array:
    logits = np.array(logits)
    y = np.array(y)

    mask = y != -100

    # Apply the mask to both logits and y
    valid_logits = logits[mask]
    valid_labels = y[mask]
    loss = mlx.nn.losses.cross_entropy(
        mx.array(valid_logits, dtype=mx.bfloat16), mx.array(valid_labels)
    ).mean()
    return loss


def loss_function(
    token_classifier: ModelForTokenClassification, x: Union[mx.array, Dict], y: mx.array
) -> mx.array:
    logits = token_classifier(x)
    return compute_loss(logits, y)


def get_validation_loss_and_metric(
    token_classifier: ModelForTokenClassification, validation_data: Dataset
) -> int:
    print("**** Computing Validation Loss ****")
    token_classifier.eval()
    total_validation_loss = 0
    for idx in range(0, len(validation_data), batch_size):
        test_slice = validation_data[idx : idx + batch_size]
        test_slice.pop("words")
        test_slice.pop("ner_labels")

        test_labels = test_slice.pop("labels")
        test_labels = mx.array([mx.array(array) for array in test_labels])
        for test_key in test_slice.keys():
            test_slice[test_key] = mx.array(
                [mx.array(array) for array in test_slice[test_key]]
            )

        total_validation_loss += loss_function(model, test_slice, test_labels).tolist()

    return total_validation_loss


if __name__ == "__main__":

    print("**** Training ****")

    train_dataset = Dataset.load_from_disk("./hf_train_ner_dataset")

    validation_dataset = Dataset.load_from_disk("./hf_test_ner_dataset")
    model: ModelForTokenClassification = ModelForTokenClassification(num_labels=21)

    loss_and_grad_fn = nn.value_and_grad(model, loss_function)

    optimizer = mlx.optimizers.Adam(learning_rate=5e-5)
    batch_size = 32
    num_epochs = 1
    epoch_iterator = tqdm(iterable=range(num_epochs), total=num_epochs, desc="Epochs")
    for epoch in epoch_iterator:
        print("Epoch :: ", epoch, " Processing!")
        train_loss_for_epoch = 0
        validation_loss_for_epoch = 0
        train_batch_iterator = tqdm(
            range(0, len(train_dataset), batch_size),
            total=len(train_dataset) // batch_size,
            desc="Batches",
        )
        model.train()
        model.bert_model.freeze()
        for i in train_batch_iterator:
            data_slice = train_dataset[i : i + batch_size]

            if "words" in data_slice.keys():
                data_slice.pop("words")
            if "ner_labels" in data_slice.keys():
                data_slice.pop("ner_labels")

            labels = data_slice.pop("labels")
            for key in data_slice.keys():
                data_slice[key] = mx.array(
                    [
                        mx.array(
                            array,
                        )
                        for array in data_slice[key]
                    ]
                )

            labels = mx.array([mx.array(array) for array in labels])
            loss_value, grads = loss_and_grad_fn(model, data_slice, labels)
            optimizer.update(model, grads)
            train_loss_for_epoch += loss_value.tolist()
        print("Training Loss :: ", train_loss_for_epoch)
        # model.eval()
        # for i in range(0, len(validation_data), batch_size):
        #     test_slice = validation_data[i: i + batch_size]
        #     test_slice.pop("words")
        #     test_slice.pop("ner_labels")

        #     test_labels = test_slice.pop("labels")
        #     test_labels = mx.array([mx.array(array) for array in test_labels])
        #     for key in test_slice.keys():
        #         test_slice[key] = mx.array(
        #             [mx.array(array) for array in test_slice[key]])

        #     validation_loss_for_epoch += loss_function(model,
        #                                                test_slice, test_labels).tolist()
        # print("validation_loss_for_epoch :: ", validation_loss_for_epoch)
        # logging.info(f"\tEpoch: {epoch} | train_loss: {
        #              train_loss_for_epoch.tolist()} | val_loss: {validation_loss_for_epoch.tolist()}")
        break

    flat_params = tree_flatten(model.parameters())
    mx.savez("pii_english_mlx_model.npz", **dict(flat_params))
