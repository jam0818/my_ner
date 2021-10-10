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


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
