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
                 special_tokens=None) -> None:
        if special_tokens is None:
            special_tokens = ['政府', '大衆']
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.special_token = special_tokens
        self.special_to_index: Dict[str, int] = {token: max_seq_len - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.sents, self.labels = self.load(path)
        # 事前に作ったものを読み込む方が安全
        label_list = get_label_list(self.labels)
        self.label_to_id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}

    def __len__(self) -> int:  # len(dataset) でデータ数を返す
        return len(self.labels)

    def __getitem__(self,
                    idx: int) -> dict[str, torch.tensor]:
        outputs = self.tokenize_and_align_labels(self.sents[idx], self.labels[idx])
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
    def tokenize_and_align_labels(self, sent, label):
        tokenized_inputs = self.tokenizer(
            sent,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        # convert str labels to int type
        input_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])
        label_ids = []
        idx = 0
        for input_token in input_tokens:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if input_token.startswith('##') or input_token in {'[CLS]', '[SEP]', '[PAD]'}:
                label_ids.append(-100)
            else:
                label_ids.append(self.label_to_id[label[idx]])
                idx += 1

        # # add special tokens
        # tokenized_inputs_special = self.tokenizer(self.special_token,
        #                                           padding=False,
        #                                           truncation=True,
        #                                           is_split_into_words=True, )
        # special_token_ids = {}
        # special_token_attention_masks = {}
        # input_special_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs_special['input_ids'])
        # for idx, input_special_token in enumerate(input_special_tokens):
        #     if not (input_special_token.startswith('##') or input_special_token in {'[CLS]', '[SEP]', '[PAD]'}):
        #         special_token_ids[input_special_token] = tokenized_inputs_special['input_ids'][idx]
        #         special_token_attention_masks[input_special_token] = tokenized_inputs_special['attention_mask'][idx]
        # for special_token in self.special_token:
        #     tokenized_inputs['input_ids'][self.special_to_index[special_token]] = special_token_ids[special_token]
        #     tokenized_inputs['attention_mask'][self.special_to_index[special_token]] = special_token_attention_masks[special_token]
        #     label_ids[self.special_to_index[special_token]] = self.label_to_id[label[self.special_to_index[special_token]]]

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs


class MyDataset2heads(Dataset):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 max_seq_len: int = 128,
                 special_tokens=None) -> None:
        if special_tokens is None:
            special_tokens = ['政府', '大衆']
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.special_token = special_tokens
        self.special_to_index: Dict[str, int] = {token: max_seq_len - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.sents, self.labels = self.load(path)
        # # 事前に作ったものを読み込む方が安全
        self.id2label = {}
        self.label_to_id = {}
        for idx, label in enumerate([[l[i] for l in self.labels] for i in range(len(self.labels[0]))]):
            label_list = get_label_list(label)
            self.label_to_id[idx] = {l: i for i, l in enumerate(label_list)}
            self.id2label[idx] = {i: l for i, l in enumerate(label_list)}

    def __len__(self) -> int:  # len(dataset) でデータ数を返す
        return len(self.labels)

    def __getitem__(self,
                    idx: int) -> dict[str, torch.tensor]:
        outputs = self.tokenize_and_align_labels(self.sents[idx], self.labels[idx])
        input_ids = torch.tensor(outputs['input_ids'])
        attention_mask = torch.tensor(outputs['attention_mask'])
        labels = torch.tensor(outputs['labels'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def load(self,
             path) -> tuple[list, list[list, list]]:
        sents = []
        labels = []
        with open(path, 'r') as f:
            for idx, line in enumerate(tqdm(f)):
                input_dict = json.loads(line)
                sent = input_dict['words']
                label = input_dict['ner']
                sents.append(sent)
                labels.append(label)

        return sents, labels

    def tokenize_and_align_labels(self, sent, label):
        tokenized_inputs = self.tokenizer(
            sent,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        label_ids_ls = []
        # convert str labels to int type
        input_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])
        for i, l in enumerate(label):
            label_ids = []
            idx = 0
            for input_token in input_tokens:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if input_token.startswith('##') or input_token in {'[CLS]', '[SEP]', '[PAD]'}:
                    label_ids.append(-100)
                else:
                    label_ids.append(self.label_to_id[i][l[idx]])
                    idx += 1

            # add special tokens
            tokenized_inputs_special = self.tokenizer(self.special_token,
                                                      padding=False,
                                                      truncation=True,
                                                      is_split_into_words=True, )
            special_token_ids = {}
            special_token_attention_masks = {}
            input_special_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs_special['input_ids'])
            for idx, input_special_token in enumerate(input_special_tokens):
                if not (input_special_token.startswith('##') or input_special_token in {'[CLS]', '[SEP]', '[PAD]'}):
                    special_token_ids[input_special_token] = tokenized_inputs_special['input_ids'][idx]
                    special_token_attention_masks[input_special_token] = tokenized_inputs_special['attention_mask'][idx]
            for special_token in self.special_token:
                tokenized_inputs['input_ids'][self.special_to_index[special_token]] = special_token_ids[special_token]
                tokenized_inputs['attention_mask'][self.special_to_index[special_token]] = special_token_attention_masks[special_token]
                if i != 0:
                    label_ids[self.special_to_index[special_token]] = self.label_to_id[i][l[self.special_to_index[special_token]]]
            label_ids_ls.append(label_ids)

        tokenized_inputs["labels"] = label_ids_ls
        return tokenized_inputs


class MyDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 dataset: Dataset,
                 max_seq_len: int = 128,
                 shuffle: bool = False,
                 batch_size: int = 2,
                 num_workers: int = 0, ) -> None:
        dataset = dataset
        self.dataset = dataset(path, tokenizer, max_seq_len)
        super(MyDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)
