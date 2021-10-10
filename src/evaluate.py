import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Union

from src.utils import align_predictions


def evaluate(model,
             dev_data_loader) -> Union[float, List]:
    with torch.no_grad():
        dev_bar = tqdm(dev_data_loader)
        predictions = []
        labels = []
        for batch_idx, batch in enumerate(dev_bar):
            batch_size = len(batch['input_ids'])
            # batch = {key: value.to(device) for key, value in batch.items()}

            # forward
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])

            loss = output.loss
            tmp_pred = torch.argmax(output.logits, dim=2).tolist()
            tmp_labels = batch['labels'].tolist()
            aligned_predictions, aligned_labels = align_predictions(tmp_pred,
                                                                    tmp_labels,
                                                                    dev_data_loader.dataset.id2label)
            predictions.extend(aligned_predictions)
            labels.extend(aligned_labels)
            score = f1_score(predictions, labels)

        return score
