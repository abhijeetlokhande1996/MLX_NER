import json
from datasets import Dataset, load_from_disk
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from utils import tokenize_and_align_labels
import torch


class ModelForTokenClassification(nn.Module):
    def __init__(self, num_labels, label2id) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.label2id = label2id
        self.bert_model = AutoModel.from_pretrained(
            "distilbert/distilbert-base-uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased")
        self.liner = nn.Linear(self.bert_model.config.dim, self.num_labels)

    def forward(self, x):

        # tokenized_inputs = tokenize_and_align_labels(
        #     examples=x, tokenizer=self.bert_tokenizer, label2id=self.label2id, return_tensors="pt")

        # for key in tokenized_inputs.keys():
        #     tokenized_inputs[key] = self.data_collator(
        #         [element for element in tokenized_inputs[key]])

        # labels = tokenized_inputs.pop("labels")
        # output = self.bert_model(**tokenized_inputs)
        pass


if __name__ == "__main__":
    with open("label2id.json", "r") as fp:
        label2id = json.load(fp)

    torch.set_default_device("mps")
    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=len(label2id), label2id=label2id)
    data_collator = DataCollatorForTokenClassification(model.bert_tokenizer)

    raw_datasets = load_from_disk("./hf_ner_dataset")

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(
            x, model.bert_tokenizer, model.label2id),
        batched=True,
        remove_columns=raw_datasets["train"].column_names)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )

    # EPOCH = 1
    # BATCH_SIZE = 8
    # for epoch in range(EPOCH):
    #     for i in range(0, len(train_dataset), BATCH_SIZE):
    #         data_slice = train_dataset[i: i + BATCH_SIZE]
    #         model(data_slice)
    #         break
