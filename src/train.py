import json
import os
from argparse import ArgumentParser
from math import log10
from pathlib import Path
from socket import gethostname

import torch
import torch.distributed as dist
from torch.nn.functional import mse_loss as l2_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup, logging

from dataset import MyDataset, MyDataLoader
from model import MyBertForTokenClassification


def train(model: MyBertForTokenClassification,
          train_data_loader: MyDataLoader,
          optimizer: AdamW,
          scheduler,
          device) -> MyBertForTokenClassification:

    total_loss = 0
    train_bar = tqdm(train_data_loader)

    for batch_idx, batch in enumerate(train_bar):
        batch_size = len(batch['input_ids'])
        batch = {key: value.to(device) for key, value in batch.items()}

        # forward
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask']).to(device)

        loss = ce_loss(output, batch['label'])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch_size

        train_bar.set_postfix({
            'lr': scheduler.get_last_lr()[0],
            'loss': round(total_loss / (batch_idx + 1), 3)
        })

    total_loss = total_loss / len(train_data_loader.dataset)
    print(f'train_loss={total_loss / (batch_idx + 1):.3f}')

    return model