import json
import logging
import typing
from abc import ABC
from typing import List, Dict

import torch
from tokenizers import Encoding
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from utils import get_label_list

logger = logging.getLogger(__file__)


class MyDataset(Dataset, ABC):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 max_seq_len: int = 128,
                 special_tokens: typing.List[str] = ['政府', '大衆']) -> None:
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.special_token = special_tokens
        self.special_to_index: Dict[str, int] = {token: max_seq_len - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.sents, self.labels = self.load(path)

    def __len__(self) -> int:  # len(dataset) でデータ数を返す
        return len(self.labels)

    def __getitem__(self,
                    idx: int) -> dict[str, torch.tensor]:
        outputs = self.tokenize_and_align_labels()
        input_ids = torch.tensor(outputs['input_ids'])
        attention_mask = torch.tensor(outputs['attention_mask'])
        labels = torch.tensor(outputs['labels'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def load(self,
             path) -> tuple[List, List]:
        sents, labels = [], []

        with open(path, 'r') as f:
            for idx, line in enumerate(tqdm(f)):
                input_dict = json.loads(line)
                sent = input_dict['words']
                label = input_dict['ner']
                sents.append(sent)
                labels.append(label)
            assert len(sents) == len(labels)
        return sents, labels

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(self):
        tokenized_inputs = self.tokenizer(
            self.sents,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        label_list = get_label_list(self.labels)
        label_to_id = {l: i for i, l in enumerate(label_list)}
        for i, label in enumerate(self.labels):
            word_ids = range(self.max_seq_len)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]])
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


class MyDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 max_seq_len: int = 128,
                 shuffle: bool = False,
                 batch_size: int = 1,
                 num_workers: int = 0, ) -> None:
        dataset = MyDataset
        self.dataset = dataset(path, tokenizer, max_seq_len)
        super(MyDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)
