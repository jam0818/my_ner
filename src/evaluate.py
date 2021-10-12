import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Union, Any

from utils import align_predictions, align_tokens


def evaluate(model,
             dev_data_loader,
             device,
             test: bool = False) -> Union[dict[str, Any], Any]:
    with torch.no_grad():
        dev_bar = tqdm(dev_data_loader)
        predictions = []
        labels = []
        if test:
            tokens = []
        for batch_idx, batch in enumerate(dev_bar):
            batch = {key: value.to(device) for key, value in batch.items()}
            tmp_tokens = []
            # forward
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])

            tmp_pred = torch.argmax(output.logits, dim=2).tolist()
            tmp_labels = batch['labels'].tolist()
            if test:
                for input_ids in batch['input_ids']:
                    tmp_tokens.append(dev_data_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids.tolist()))
                tokens.extend(align_tokens(tmp_tokens, tmp_labels))
            aligned_predictions, aligned_labels = align_predictions(tmp_pred,
                                                                    tmp_labels,
                                                                    dev_data_loader.dataset.id2label)
            predictions.extend(aligned_predictions)
            labels.extend(aligned_labels)
        score = f1_score(predictions, labels)
        if not test:
            return score
        else:
            return {
                'score': score,
                'tokens': tokens,
                'labels': labels,
                'prediction': predictions
            }
