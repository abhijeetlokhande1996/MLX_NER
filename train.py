import json
import mlx.core
import numpy as np

import mlx.core as mx
from mlx.core import array


from model import load_model
from datasets import Dataset, load_from_disk
from transformers import DataCollatorForTokenClassification, get_scheduler
from torch.utils.data import DataLoader
import mlx
from mlx.optimizers import Adam
from tqdm import tqdm

import logging
import mlx.nn as nn
from mlx.utils import tree_flatten
from typing import Dict, Union, Any, Tuple
from sklearn.metrics import f1_score
from utils import tokenize_and_align_labels

logging.basicConfig(
    level=logging.INFO,
    filename="mlx_train.log",
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ModelForTokenClassification(nn.Module):
    def __init__(self, num_labels, label2id, max_length=512) -> None:
        super().__init__()
        self.bert_model, self.bert_tokenizer = load_model(
            "bert-base-uncased", "weights/bert-base-uncased.npz"
        )
        self.bert_model.freeze()
        self.bert_model_output_dim = 768
        self.label2id = label2id
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, num_labels)
        self.label_weights = mx.array(
            [
                0.1,  # O
                1.0,  # B-FIRSTNAME
                1.0,  # I-FIRSTNAME
                1.0,  # B-MIDDLENAME
                1.0,  # I-MIDDLENAME
                1.0,  # B-LASTNAME
                1.0,  # I-LASTNAME
                1.0,  # B-SSN
                1.0,  # I-SSN
                1.0,  # B-ACCOUNTNUMBER
                1.0,  # I-ACCOUNTNUMBER
                1.0,  # B-CREDITCARDNUMBER
                1.0,  # I-CREDITCARDNUMBER
                1.0,  # B-DOB
                1.0,  # I-DOB
                1.0,  # B-EMAIL
                1.0,  # I-EMAIL
                1.0,  # B-PASSWORD
                1.0,  # I-PASSWORD
                1.0,  # B-PHONENUMBER
                1.0,  # I-PHONENUMBER
            ]
        )

    def __call__(self, x: Dict) -> mx.array:
        # labels = mx.array(x.pop("labels"))
        tokens = {key: mx.array(v) for key, v in x.items()}

        labels = tokens.pop("labels")
        output, _ = self.bert_model(**tokens)
        output = self.dropout(output)
        output = self.linear1(output)
        logits = mx.softmax(output, axis=-1)

        return logits, labels


def compute_loss(logits: mx.array, y: mx.array, weights: mx.array) -> array:
    import pdb
    pdb.set_trace()
    mask = np.array(y != -100).astype("int")
    logits = logits * mask
    loss = nn.losses.cross_entropy(logits=logits, targets=y, reduction="mean")
    return loss


def loss_function(logits: mx.array, y: mx.array, weights: mx.array) -> mx.array:
    return compute_loss(logits, y, weights)


def calculate_f1_score(logits: mx.array, y: mx.array) -> float:
    logits = np.array(logits)  # type: ignore
    y = np.array(y)  # type: ignore
    mask = y != -100
    valid_logits = logits[mask]
    valid_labels = y[mask]
    predicted = np.argmax(valid_logits, axis=1)
    f1 = f1_score(valid_labels, predicted, average="micro")
    return f1


if __name__ == "__main__":

    print("**** Training ****")
    with open("label2id.json", "r") as fp:
        label2id = json.load(fp)

    NUM_LABELS = len(label2id)
    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=NUM_LABELS, label2id=label2id
    )

    # raw_datasets = load_from_disk("./hf_ner_dataset")
    # tokenized_datasets = raw_datasets.map(
    #     lambda x: tokenize_and_align_labels(
    #         x, model.bert_tokenizer, model.label2id, max_length=200, return_tensors="np",
    #     ),
    #     batched=True,
    # )
    # tokenized_datasets.save_to_disk("./hf_ner_tokenized_dataset")

    tokenized_datasets = load_from_disk("./hf_ner_tokenized_dataset")
    train_tokenized_datasets = tokenized_datasets["train"]
    test_tokenized_datasets = tokenized_datasets["test"]

    loss_and_grad_fn = nn.value_and_grad(model, loss_function)

    optimizer = Adam(learning_rate=5e-5)

    batch_size = 32
    num_epochs = 3

    epoch_iterator = tqdm(iterable=range(num_epochs),
                          total=num_epochs, desc="Epochs")

    for epoch in epoch_iterator:
        train_loss_for_epoch: float = 0
        validation_loss_for_epoch = 0
        train_batch_iterator = tqdm(range(0, len(train_tokenized_datasets), batch_size), total=len(train_tokenized_datasets) // batch_size,
                                    desc="Batches",
                                    )
        model.train()
        for i in train_batch_iterator:
            data_slice = train_tokenized_datasets[i: i + batch_size]

            if "words" in data_slice:
                data_slice.pop("words")
            if "ner_labels" in data_slice:
                true_labels = data_slice.pop("ner_labels")
            import pdb
            pdb.set_trace()
            logits, labels = model(data_slice)

            loss_value, grads = loss_and_grad_fn(
                logits, labels, model.label_weights)
            optimizer.update(model, grads)
            train_loss_for_epoch += loss_value.tolist()
            f1 = calculate_f1_score(logits, labels)
            logging.info(
                f"Epoch: {epoch}\tBatch: {
                    i//batch_size}\tLoss: {loss_value.tolist()}\t\t\t\tF1: {f1}"
            )
            break
        logging.info(f"Epoch: {epoch}\t\tLoss: {train_loss_for_epoch}")

        flat_params = tree_flatten(model.parameters())
        mx.savez(
            f"./models/pii_english_mlx_model_epoch_{epoch}.npz", **dict(flat_params)
        )
