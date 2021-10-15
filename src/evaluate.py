import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Union, Any

from utils import align_predictions, align_tokens


def evaluate(model,
             dev_data_loader,
             device,) -> Union[dict[str, Any], Any]:
    with torch.no_grad():
        dev_bar = tqdm(dev_data_loader)
        id2k = {0: 'topic', 1: 'target'}
        predictions = {'topic': [], 'target': []}
        labels = {'topic': [], 'target': []}
        score = {'topic': None, 'target': None}

        for batch_idx, batch in enumerate(dev_bar):
            batch = {key: value.to(device) for key, value in batch.items()}
            # forward
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])

            for idx in id2k.keys():
                tmp_pred = torch.argmax(output.logits[id2k[idx]], dim=2).tolist()
                tmp_labels = batch['labels'][:, idx].tolist()
                aligned_predictions, aligned_labels = align_predictions(tmp_pred,
                                                                        tmp_labels,
                                                                        dev_data_loader.dataset.id2label[idx])
                predictions[id2k[idx]].extend(aligned_predictions)
                labels[id2k[idx]].extend(aligned_labels)
                score[id2k[idx]] = f1_score(predictions[id2k[idx]], labels[id2k[idx]])
        return {
            'score': score,
            'labels': labels,
            'prediction': predictions
        }
