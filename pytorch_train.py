import json
from datasets import load_from_disk
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import DataCollatorForTokenClassification, get_scheduler
from torch.utils.data import DataLoader
from utils import tokenize_and_align_labels
import torch
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict
import logging
import evaluate
import os
from seqeval.metrics import f1_score, precision_score, recall_score

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


logging.basicConfig(
    level=logging.INFO,
    filename="pytorch_train.log",
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ModelForTokenClassification(nn.Module):
    def __init__(self, num_labels, label2id) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.bert_model = AutoModel.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.liner1 = nn.Linear(self.bert_model.config.dim, self.num_labels)
        # self.freeze_bert_model()

    def freeze_bert_model(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.bert_model(**x)
        output = self.liner1(output.last_hidden_state)
        return output


def post_process(predictions, labels, id2label):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_labels = [[id2label[gt] for gt in label if gt != -100]
                   for label in labels]

    true_predictions = [
        [id2label[p] for p, gt in zip(pred, label) if gt != -100]
        for pred, label in zip(predictions.argmax(axis=-1), labels)
    ]
    return true_labels, true_predictions


if __name__ == "__main__":
    with open("label2id.json", "r") as fp:
        label2id = json.load(fp)

    metric = evaluate.load("seqeval")

    BATCH_SIZE = 8
    torch.set_default_device("mps")
    # torch.set_default_dtype(torch.float16)

    device = torch.device("mps")

    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=len(label2id), label2id=label2id
    )
    model.to(device)
    data_collator = DataCollatorForTokenClassification(model.bert_tokenizer)

    raw_datasets = load_from_disk("./hf_ner_dataset")

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(
            x, model.bert_tokenizer, model.label2id),
        batched=True,
        remove_columns=["words", "ner_labels"],
    )

    # Create a generator for the DataLoader
    generator = torch.Generator(device=device)
    generator.manual_seed(42)  # Set a fixed seed for reproducibility

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        generator=generator,
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        generator=generator,
    )

    num_train_epochs = 5
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))
    weights = torch.tensor(
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
    assert len(weights) == len(label2id)
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100, reduction="mean", weight=weights)
    for train_step in range(num_train_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # Ensure all tensors in batch are on the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").long()

            logits = model(batch)
            loss = criterion(
                logits.view(-1, logits.shape[-1]), labels.view(-1))

            true_labels, true_predictions = post_process(
                logits, labels, model.id2label)

            # metric.add_batch(predictions=true_predictions, references=true_labels)
            precision = precision_score(
                true_predictions, true_labels, average="weighted"
            )
            recall = recall_score(
                true_predictions, true_labels, average="weighted")
            f1 = f1_score(true_predictions, true_labels, average="weighted")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

            info_string = f"Train Step: {train_step}\t\tBatch: {batch_idx}\t\tLoss: {
                loss.item()}\t\tPrecision: {precision*100}\t\tRecall: {recall*100}\t\tF1: {f1*100}"
            logging.info(info_string)
        logging.info("--" * 100)
    try:
        torch.save(model.state_dict(), './models/model.pth')
    except Exception as e:
        torch.save(model.state_dict(), './model.pth')

    # model.eval()

    # for test_step, batch in enumerate(test_dataloader):
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     labels = batch.pop("labels").long()

    #     with torch.no_grad():
    #         logits = model(batch)

    #     true_labels, true_predictions = post_process(
    #         logits, labels, model.id2label)

    #     # metric.add_batch(predictions=true_predictions, references=true_labels)
    #     precision = precision_score(
    #         true_predictions, true_labels, average="weighted"
    #     )
    #     recall = recall_score(
    #         true_predictions, true_labels, average="weighted")
    #     f1 = f1_score(true_predictions, true_labels, average="weighted")

    #     info_string = f"Test Step: {test_step}\t\tBatch: {batch_idx}\t\tPrecision: {
    #         precision*100}\t\tRecall: {recall*100}\t\tF1: {f1*100}"
    #     logging.info(info_string)
