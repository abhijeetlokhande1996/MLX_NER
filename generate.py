import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import List, Dict
from sklearn.model_selection import train_test_split
from utils import tokenize_and_align_labels
from model import Bert, load_model
from pprint import pprint
from typing import List, Tuple

FIRST_ORDER_PII_ENTITIES = [
    "O",
    "FIRSTNAME",
    "MIDDLENAME",
    "LASTNAME",
    "SSN",
    "ACCOUNTNUMBER",
    "CREDITCARDNUMBER",
    "DOB",
    "EMAIL",
    "PASSWORD",
    "PHONENUMBER",]


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
        entity: False for entity in FIRST_ORDER_PII_ENTITIES if entity != "O"
    }
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


def replace_entities_with_other_if_not_present(
    data: List[Dict], labels=None
) -> List[Dict]:
    if labels is None:
        labels = FIRST_ORDER_PII_ENTITIES
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


def generate_tokenised_dataset(json_file_path: Path) -> Tuple[Dataset, Dataset, Dataset]:
    with open(json_file_path, "r") as fp:
        data = json.load(fp)

    # First split the data into training and temporary sets
    train_data, temp_data = train_test_split(
        data, test_size=0.2, random_state=42, shuffle=True)

    # Then split the temporary set into validation and test sets
    validation_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, shuffle=True)

    if not check_first_order_pii_is_present(train_data):
        raise Exception("First order PII entities are not present")

    train_data = replace_entities_with_other_if_not_present(train_data)

    test_data = replace_entities_with_other_if_not_present(test_data)

    validation_data = replace_entities_with_other_if_not_present(
        validation_data)

    label_id = 0
    label2id = {"O": label_id}
    for entity in FIRST_ORDER_PII_ENTITIES:
        if entity != "O":
            label_id += 1
            label2id[f"B-{entity}"] = label_id
            label_id += 1
            label2id[f"I-{entity}"] = label_id

    label2id = dict(sorted(label2id.items(), key=lambda item: item[1]))
    with open("label2id.json", "w") as fp:
        json.dump(label2id, fp, indent=3)
        print("*** Label2Id dumped ***")
    print("Num labels", len(label2id))
    train_hf_dataset = Dataset.from_list(train_data)
    test_hf_dataset = Dataset.from_list(test_data)
    validation_hf_dataset = Dataset.from_list(validation_data)

    # bert_model, bert_tokenizer = load_model(
    #     "bert-base-uncased", "weights/bert-base-uncased.npz"
    # )

    # train_hf_dataset = train_hf_dataset.map(
    #     lambda batch: tokenize_and_align_labels(batch, bert_tokenizer, label2id),
    #     batched=True,
    #     batch_size=32,
    # )

    # test_hf_dataset = test_hf_dataset.map(
    #     lambda batch: tokenize_and_align_labels(batch, bert_tokenizer, label2id),
    #     batched=True,
    #     batch_size=32,
    # )
    return (train_hf_dataset, test_hf_dataset, validation_hf_dataset)


if __name__ == "__main__":
    print("*** Generating Dataset ***")
    json_path = Path(".").resolve() / "pii_training_data.json"
    train_dataset, test_dataset, validation_hf_dataset = generate_tokenised_dataset(
        json_file_path=json_path)
    DatasetDict({"train": train_dataset, "test": test_dataset,
                "validation": validation_hf_dataset}).save_to_disk("./hf_ner_dataset")
    print("*** Done ***")
