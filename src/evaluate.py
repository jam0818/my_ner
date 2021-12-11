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
        id2k = dev_data_loader.dataset.id2key
        predictions = {'topic': [], 'target': []}
        labels = {'topic': [], 'target': []}
        score = {'topic': None, 'target': None}

        for batch_idx, batch in enumerate(dev_bar):
            batch = {key: value.to(device) for key, value in batch.items()}
            # forward
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           token_type_ids=batch['token_type_ids'],
                           labels=batch['labels'])

            for idx, key in id2k.items():
                tmp_pred = torch.argmax(output.logits[key], dim=2).tolist()
                tmp_labels = batch['labels'][:, idx].tolist()
                aligned_predictions, aligned_labels = align_predictions(tmp_pred,
                                                                        tmp_labels,
                                                                        dev_data_loader.dataset.id2label[key])
                predictions[key].extend(aligned_predictions)
                labels[key].extend(aligned_labels)
                score[key] = f1_score(predictions[key], labels[key])
        return {
            'score': score,
            'labels': labels,
            'prediction': predictions
        }
