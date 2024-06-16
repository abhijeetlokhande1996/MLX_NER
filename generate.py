import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import List, Dict
from sklearn.model_selection import train_test_split
from utils import tokenize_and_align_labels
from model import Bert, load_model
from pprint import pprint

FIRST_ORDER_PII_ENTITIES = ("O", "FIRSTNAME", "MIDDLENAME", "LASTNAME",
                            "SSN", "ACCOUNTNUMBER", "CREDITCARDNUMBER",
                            "DOB", "EMAIL", "PASSWORD", "PHONENUMBER", "EMAIL")


def get_label2id(data: List[Dict]) -> dict:
    unique_labels = set()
    for item in data:
        for label in item["ner_labels"]:
            unique_labels.add(label)
    pprint(unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    return label2id


def check_first_order_pii_is_present(data: List[Dict]) -> bool:
    entity_dict = {
        entity: False for entity in FIRST_ORDER_PII_ENTITIES if entity != "O"}
    for item in data:
        for ner_label in item["ner_labels"]:
            if ner_label == "O":
                continue
            label = ner_label.split("-")[1]
            if label in FIRST_ORDER_PII_ENTITIES:
                if entity_dict[label]:
                    continue
                entity_dict[label] = True
    return all(entity_dict.values())


def replace_entities_with_other_if_not_present(data: List[Dict], labels: List[str]) -> List[Dict]:
    replaced_label = set()
    for item in data:
        for i, ner_label in enumerate(item["ner_labels"]):
            if ner_label == "O":
                continue
            label = ner_label.split("-")[1]
            if not (label in labels):
                replaced_label.add(label)
                item["ner_labels"][i] = "O"

    return data


def generate_tokenised_dataset(json_path: Path) -> None:
    with open(json_path, "r") as fp:
        data = json.load(fp)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, shuffle=True)

    if not check_first_order_pii_is_present(train_data):
        raise Exception("First order PII entities are not present")

    train_data = replace_entities_with_other_if_not_present(
        train_data, FIRST_ORDER_PII_ENTITIES)

    test_data = replace_entities_with_other_if_not_present(
        test_data, FIRST_ORDER_PII_ENTITIES)

    label_id = 0
    label2id = {"O": label_id}
    for entity in FIRST_ORDER_PII_ENTITIES:
        if entity != "O":
            label_id += 1
            label2id[f"B-{entity}"] = label_id
            label_id += 1
            label2id[f"I-{entity}"] = label_id

    print(label2id)
    print("Num labels", len(label2id))
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    bert_model, bert_tokenizer = load_model(
        "bert-base-uncased", "weights/bert-base-uncased.npz")

    train_dataset = train_dataset.map(
        lambda batch: tokenize_and_align_labels(
            batch, bert_tokenizer, label2id),
        batched=True,
        batch_size=32,
    )

    test_dataset = test_dataset.map(
        lambda batch: tokenize_and_align_labels(
            batch, bert_tokenizer, label2id),
        batched=True,
        batch_size=32,
    )
    return (train_dataset, test_dataset)


if __name__ == "__main__":
    json_path = Path(".").resolve().parent / "bert" / "pii_training_data.json"
    train_dataset, test_dataset = generate_tokenised_dataset(
        json_path=json_path)
    train_dataset.save_to_disk("hf_train_ner_dataset")
    test_dataset.save_to_disk("hf_test_ner_dataset")
    print("*** Done ***")
