import dataclasses
import json
import os
import pathlib
import re
from datetime import datetime
from random import random

import numpy as np
import torch
import typing

from tqdm import tqdm


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpuid: str) -> torch.device:
    if gpuid and torch.cuda.is_available():
        assert re.fullmatch(r'[0-7]', gpuid) is not None, 'invalid way to specify gpuid'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        device = torch.device(f'cuda:{gpuid}')
    else:
        device = torch.device('cpu')

    return device


@dataclasses.dataclass
class Tweet:
    raw_text: str  # Raw text.
    text: str  # Raw text without mentions, links, and hashtags.
    sents: typing.List[str]  # A list of the sentences (mentions, links, and hashtags are removed).
    tokenized_sents: typing.List[str]  # A list of the tokenized sentences (mentions, links, and hashtags are removed).
    content_words: typing.List[str]  # A list of the content words.
    mentions: typing.List[str]  # The list of mentions removed from the raw text.
    links: typing.List[str]  # The list of links removed from the raw text.
    hashtags: typing.List[str]  # The list of hashtags removed from the raw text.
    _is_quote: bool  # if True, the tweet is retweet
    timestamp: str  # Timestamp.
    lang: str  # Language
    created_from: str  # Path to the raw data.
    datetime: typing.Optional[datetime] = None
    date: typing.Optional[datetime] = None

    def __post_init__(self):
        self.datetime = datetime.strptime(self.timestamp, "%a %b %d %H:%M:%S +0000 %Y")
        self.date = self.datetime.date()

    def to_dict(self):
        exclude = {"datetime", "date"}
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self) if field.name not in exclude
        }


def load_tweets(path: str) -> typing.List[Tweet]:
    tweets = []
    for path in tqdm(pathlib.Path(path).glob("**/*.jsonl")):
        with path.open("rt") as f:
            for line in f:
                if line.strip() == "":
                    continue
                tweet = Tweet(**json.loads(line))
                tweets.append(tweet)
    return tweets


def save_df_to_pkl(df, output_filename):
    file_path = pathlib.Path(output_filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(str(file_path))
