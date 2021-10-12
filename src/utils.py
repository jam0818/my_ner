import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score



def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def align_predictions(predictions, labels, id2label) -> tuple[list, list]:
    batch_size = len(predictions)
    max_seq_len = len(predictions[0])
    IGNORE_INDEX = -100

    aligned_predictions = [[] for _ in range(batch_size)]
    aligned_labels = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(max_seq_len):
            if labels[i][j] != IGNORE_INDEX:
                aligned_predictions[i].append(id2label[predictions[i][j]])
                aligned_labels[i].append(id2label[labels[i][j]])

    return aligned_predictions, aligned_labels


def align_tokens(tokens, labels) -> list[list]:
    batch_size = len(labels)
    max_seq_len = len(labels[0])
    IGNORE_INDEX = -100

    aligned_tokens = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(max_seq_len):
            if labels[i][j] != IGNORE_INDEX:
                aligned_tokens[i].append(tokens[i][j])

    return aligned_tokens

