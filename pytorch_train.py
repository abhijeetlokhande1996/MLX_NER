import json
from datasets import Dataset, load_from_disk
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, get_scheduler
from torch.utils.data import DataLoader
from utils import tokenize_and_align_labels
import torch
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict
import logging


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
        self.bert_model = AutoModel.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.liner1 = nn.Linear(self.bert_model.config.dim, self.num_labels)
        self.freeze_bert_model()

    def freeze_bert_model(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.bert_model(**x)
        output = self.liner1(output.last_hidden_state)
        return output

        pass


if __name__ == "__main__":
    with open("label2id.json", "r") as fp:
        label2id = json.load(fp)

    BATCH_SIZE = 16
    torch.set_default_device("mps")
    device = torch.device("mps")

    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=len(label2id), label2id=label2id
    )
    model.to(device)
    data_collator = DataCollatorForTokenClassification(model.bert_tokenizer)

    raw_datasets = load_from_disk("./hf_ner_dataset")

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, model.bert_tokenizer, model.label2id),
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

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    optimizer = AdamW(model.parameters(), lr=2e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    for i in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # Ensure all tensors in batch are on the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").long()

            logits = model(batch)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

            break
        break

        # print(batch.shape)
