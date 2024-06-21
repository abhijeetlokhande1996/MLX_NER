import json
from datasets import Dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification

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
        self.data_collator = DataCollatorForTokenClassification(
            self.bert_tokenizer)
        self.liner = nn.Linear(self.bert_model.config.dim, self.num_labels)

    def forward(self, x):
        tokenized_inputs = tokenize_and_align_labels(
            examples=x, tokenizer=self.bert_tokenizer, label2id=self.label2id, return_tensors="pt")

        labels = tokenized_inputs.pop("labels")

        print(tokenized_inputs.keys())
        output = self.bert_model(**tokenized_inputs)
        print("Shape :: ", output.last_hidden_state.shape)
        pass


if __name__ == "__main__":
    with open("label2id.json", "r") as fp:
        label2id = json.load(fp)

    torch.set_default_device("mps")
    model: ModelForTokenClassification = ModelForTokenClassification(
        num_labels=len(label2id), label2id=label2id)

    train_dataset = Dataset.load_from_disk("./hf_train_ner_dataset")

    validation_dataset = Dataset.load_from_disk("./hf_test_ner_dataset")

    EPOCH = 1
    BATCH_SIZE = 8
    for epoch in range(EPOCH):
        for i in range(0, len(train_dataset), BATCH_SIZE):
            data_slice = train_dataset[i: i + BATCH_SIZE]
            model(data_slice)
            break
