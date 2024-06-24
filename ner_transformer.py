import torch
import torch.nn as nn
from transformers import BertTokenizer
from datasets import load_from_disk, DatasetDict
from typing import Union, Dict, List


class NERTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int) -> None:
        super(NERTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.tensor, float]:
        return 0.0


if __name__ == "__main__":
    ner_dataset: DatasetDict = load_from_disk("hf_ner_dataset")
    train_dataset = ner_dataset["train"]
    test_dataset = ner_dataset["test"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_inputs: Dict[str, torch.Tensor] = tokenizer(
        train_dataset["words"], padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)

    input_ids = tokenized_inputs.pop("input_ids")
    attention_mask = tokenized_inputs.pop("attention_mask")
