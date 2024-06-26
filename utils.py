def align_labels_with_tokens(labels, word_ids, text_labels):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            # TODO: need to check
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples, tokenizer, label2id,  max_length=None, return_tensors="np"):
    if not max_length:
        tokenized_inputs = tokenizer(
            examples["words"],
            is_split_into_words=True,
            truncation=True,
            return_tensors=return_tensors,
        )
    else:
        tokenized_inputs = tokenizer(
            examples["words"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=return_tensors,
        )

    all_labels = []
    all_labels_names = []

    for labels in examples["ner_labels"]:
        ner_labels = []
        temp_labels_names = []
        for label in labels:
            ner_labels.append(label2id[label])
            temp_labels_names.append(label)
        all_labels.append(ner_labels)
        all_labels_names.append(temp_labels_names)

    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(
            align_labels_with_tokens(labels, word_ids, all_labels_names[i])
        )

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
