import json

if __name__ == "__main__":
    with open("pii_training_data.json", "r") as fp:
        data = json.load(fp)

    data_to_write = [None] * len(data)
    for idx, sample in enumerate(data):
        text = f"<|user|>\n Extract entities and format as JSON: {' '.join(sample["words"])}<|end|>\n"
        text += f"<|assistant|>\n {json.dumps(sample["ner_labels"])}<|end|>"
        data_to_write[idx] = {
            "text": text
        }
    
    with open("./phi3_input.json", "w") as fp:
        json.dump(data_to_write, fp, indent=4)
        print("**Done**")
    

