from label_name_map import semantic_label_name_map

def mask_dataset(dataset, label_column, label_semantic_level, labels_to_keep):
    tag_info = dataset["train"].features[label_column]
    dataset_name = dataset["train"].info.dataset_name
    labels_for_model = {}
    labels_for_masking = {}
    curr_idx = 0
    for old_label_idx, old_label in enumerate(dataset["train"].features[label_column].feature.names):
        if any([old_label.startswith(l) for l in labels_to_keep]) or old_label == "O":
            labels_for_masking[old_label_idx] = curr_idx
            labels_for_model[curr_idx] = semantic_label_name_map[dataset_name][f"{label_column}_{label_semantic_level}"][
                dataset["train"].features[label_column].feature.names[old_label_idx]
            ]
            curr_idx += 1
        else:
            labels_for_masking[old_label_idx] = 0

    def mask(examples):
        examples[label_column] = [[labels_for_masking.get(old_id) for old_id in sample] for sample in
                                     examples[label_column]]
        return examples

    dataset = dataset.map(mask, batched=True)

    tag_info.feature.names = list(labels_for_model.values())
    features = dataset["train"].features
    features[label_column] = tag_info

    dataset = dataset.cast(features)

    return dataset

def mask_full_dataset(dataset, label_column, label_semantic_level, pretraining_labels):
    tag_info_train = dataset["train"].features[label_column]
    tag_info_eval = dataset["validation"].features[label_column]

    dataset_name = dataset["train"].info.dataset_name

    train_id2label = {0: "outside"}
    eval_id2label = {0: "outside"}
    train_split_mask = {0: 0}
    eval_split_mask = {0: 0}
    new_idx_for_train = 1
    new_idx_for_eval = 1

    for original_label_idx, original_label in enumerate(dataset["train"].features[label_column].feature.names):
        if original_label == "O":
            continue
        if any([original_label.startswith(label_for_pretraining) for label_for_pretraining in pretraining_labels]):
            train_split_mask[original_label_idx] = new_idx_for_train
            eval_split_mask[original_label_idx] = 0

            train_id2label[new_idx_for_train] = semantic_label_name_map.get(dataset_name).get(
                f"{label_column}_{label_semantic_level}").get(
                dataset["train"].features[label_column].feature.names[original_label_idx])

            new_idx_for_train += 1
        else:
            train_split_mask[original_label_idx] = 0
            eval_split_mask[original_label_idx] = new_idx_for_eval

            eval_id2label[new_idx_for_eval] = semantic_label_name_map.get(dataset_name).get(
                f"{label_column}_{label_semantic_level}").get(
                dataset["train"].features[label_column].feature.names[original_label_idx])

            new_idx_for_eval += 1

    def mask_train(examples):
        examples[label_column] = [[train_split_mask.get(old_id) for old_id in sample] for sample in
                                     examples[label_column]]
        return examples

    def mask_eval(examples):
        examples[label_column] = [[eval_split_mask.get(old_id) for old_id in sample] for sample in
                                     examples[label_column]]
        return examples

    dataset["train"] = dataset["train"].map(mask_train, batched=True)
    dataset["validation"] = dataset["validation"].map(mask_eval, batched=True)
    dataset["test"] = dataset["test"].map(mask_eval, batched=True)

    tag_info_train.feature.names = list(train_id2label.values())
    tag_info_eval.feature.names = list(eval_id2label.values())

    train_features = dataset["train"].features
    train_features[label_column] = tag_info_train
    dataset["train"] = dataset["train"].cast(train_features)

    eval_features = dataset["validation"].features
    eval_features[label_column] = tag_info_eval
    dataset["validation"] = dataset["validation"].cast(eval_features)

    test_features = dataset["test"].features
    test_features[label_column] = tag_info_eval
    dataset["test"] = dataset["test"].cast(test_features)

    return dataset
