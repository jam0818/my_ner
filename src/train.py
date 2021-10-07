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

from dataset import MyDataset
from model import BertForTokenClassification


def tarin(model: BertForTokenClassification,
          train_data_loader: )
