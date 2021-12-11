import dataclasses
import datetime
import json
import logging
import re
import typing
from abc import ABC
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertTokenizer

from utils import Tweet, load_tweets
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

        ## add special tokens
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
                 special_tokens=None,
                 tag_type='BIO') -> None:
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.special_token = special_tokens
        if self.special_token is not None:
            self.special_to_index: Dict[str, int] = {token: max_seq_len - i - 1 for i, token
                                                     in enumerate(reversed(special_tokens))}
        self.sents, self.labels = self.load(path)
        # # 事前に作ったものを読み込む方が安全
        self.key2id = {'topic': 0, 'target': 1}
        self.id2key = {v: k for k, v in self.key2id.items()}
        if tag_type == 'BIO':
            self.id2label = {'topic': {0: 'B-TOPIC', 1: 'I-TOPIC', 2: 'O'},
                             'target': {0: 'B-TARGET', 1: 'I-TARGET', 2: 'O'}}
            self.label_to_id = {'topic': {'B-TOPIC': 0, 'I-TOPIC': 1, 'O': 2},
                                'target': {'B-TARGET': 0, 'I-TARGET': 1, 'O': 2}}
        elif tag_type == 'span':
            self.id2label = {'topic': {0: 'I-TOPIC', 1: 'O'},
                             'target': {0: 'I-TARGET', 1: 'O'}}
            self.label_to_id = {'topic': {'I-TOPIC': 0, 'O': 1},
                                'target': {'I-TARGET': 0, 'O': 1}}

    def __len__(self) -> int:  # len(dataset) でデータ数を返す
        return len(self.labels)

    def __getitem__(self,
                    idx: int) -> dict[str, torch.tensor]:
        outputs = self.tokenize_and_align_labels(self.sents[idx], self.labels[idx])
        input_ids = torch.tensor(outputs['input_ids'])
        attention_mask = torch.tensor(outputs['attention_mask'])
        token_type_ids = torch.tensor(outputs['token_type_ids'])
        labels = torch.tensor(outputs['labels'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
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
        label_ids_ls = [[] for _ in range(len(self.key2id))]
        label_dic = {'topic': label[self.key2id['topic']],
                     'target': label[self.key2id['target']]}
        # convert str labels to int type
        input_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'])
        for k, v in label_dic.items():
            label_ids = []
            idx = 0
            for input_token in input_tokens:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if input_token.startswith('##') or input_token in {'[CLS]', '[SEP]', '[PAD]'}:
                    label_ids.append(-100)
                else:
                    label_ids.append(self.label_to_id[k][v[idx]])
                    idx += 1

            # add special tokens
            if self.special_token is not None:
                special_token_ids = {}
                indexes = self.tokenizer.convert_tokens_to_ids(self.special_token)
                for idx, special_token in zip(indexes, self.special_token):
                    special_token_ids[special_token] = idx
                for special_token in self.special_token:
                    tokenized_inputs['input_ids'][self.special_to_index[special_token]] = special_token_ids[
                        special_token]
                    tokenized_inputs['attention_mask'][self.special_to_index[special_token]] = 1
                    if k == 'target':
                        label_ids[self.special_to_index[special_token]] = self.label_to_id[k][
                            v[self.special_to_index[special_token]]]

            label_ids_ls[self.key2id[k]] = label_ids

        tokenized_inputs["labels"] = label_ids_ls
        return tokenized_inputs


class MyDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 dataset: Dataset,
                 max_seq_len: int = 128,
                 special_tokens=None,
                 shuffle: bool = False,
                 batch_size: int = 2,
                 num_workers: int = 0,
                 tag_type='BIO') -> None:
        self.dataset = dataset(path,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len,
                               special_tokens=special_tokens,
                               tag_type=tag_type)
        super(MyDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)


class TwitterDataset(Dataset):
    """path 以下にある jsonl ファイルたちを読み込むクラス．"""

    def __init__(
            self,
            path: str,
            model_name_or_path: str,
            max_seq_length: int = 128,
            special_tokens=None,
            filter_fns: typing.Optional[typing.List[typing.Callable[[Tweet], bool]]] = None,
    ):
        # filter_fns をすべてクリアしたツイートだけ使う
        self.tweets = load_tweets(path)
        if filter_fns is not None:
            self.tweets = [t for t in self.tweets if not any(fn(t) for fn in filter_fns)]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False, tokenize_chinese_chars=False
        )
        self.max_seq_length = max_seq_length
        self.special_token = special_tokens
        if self.special_token is not None:
            self.special_to_index: Dict[str, int] = {token: max_seq_length - i - 1 for i, token
                                                     in enumerate(reversed(special_tokens))}
        self.key2id = {'topic': 0, 'target': 1}
        self.id2key = {v: k for k, v in self.key2id.items()}
        self.id2label = {'topic': {0: 'B-TOPIC', 1: 'I-TOPIC', 2: 'O'},
                         'target': {0: 'B-TARGET', 1: 'I-TARGET', 2: 'O'}}
        self.label_to_id = {'topic': {'B-TOPIC': 0, 'I-TOPIC': 1, 'O': 2},
                            'target': {'B-TARGET': 0, 'I-TARGET': 1, 'O': 2}}

    def __getitem__(self, idx: int):
        return self.convert_example_to_features(self.tweets[idx])

    def __len__(self):
        return len(self.tweets)

    def convert_example_to_features(self, tweet: Tweet) -> dict:
        input_tokens = [self.tokenizer.cls_token]
        attention_mask = [1]
        token_type_ids = [0]
        for i, tokenized_sent in enumerate(tweet.tokenized_sents):
            cur_input_tokens = self.tokenizer.tokenize(tokenized_sent) + [self.tokenizer.sep_token]
            input_tokens += cur_input_tokens
            attention_mask += [1] * len(cur_input_tokens)
            token_type_ids += [i % 2] * len(cur_input_tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        # Truncate tokens when the sequence is too long.
        input_ids = input_ids[: self.max_seq_length]
        attention_mask = attention_mask[: self.max_seq_length]
        token_type_ids = token_type_ids[: self.max_seq_length]
        # Add padding tokens for batch processing.
        seq_length = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - seq_length)
        attention_mask += [0] * (self.max_seq_length - seq_length)
        token_type_ids += [0] * (self.max_seq_length - seq_length)
        # Add special tokens
        label = [[] for _ in range(len(self.key2id))]
        labels_ls = [[] for _ in range(len(self.key2id))]
        label_dic = {'topic': [0],
                     'target': label[self.key2id['target']]}
        # convert str labels to int type
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        for k, v in label_dic.items():
            label_ids = []
            idx = 0
            for input_token in input_tokens:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if input_token.startswith('##') or input_token in {'[CLS]', '[SEP]', '[PAD]'}:
                    label_ids.append(-100)
                else:
                    label_ids.append(0)
                    idx += 1
            labels_ls[self.key2id[k]] = label_ids
        for s_token, i in self.special_to_index.items():
            input_ids[i] = self.tokenizer.convert_tokens_to_ids(s_token)
            attention_mask[i] = 1
            # targetのみ特殊トークンを見る
            labels_ls[self.key2id['target']][i] = 0

        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels_ls
        }
        features.update(dataclasses.asdict(tweet))
        return features

def filter_by_timestamp(from_: str = "", to: str = "") -> typing.Callable[[Tweet], bool]:
    """Filter out tweets by their timestamp.

    Args:
        from_ (str): Format is 'yyyy/mm/dd'.
        to (str): Format is 'yyyy/mm/dd'.

    Returns:
        True if this tweet is to be filtered out. Otherwise, False.
    """
    from_ = datetime.date(*(map(int, from_.split("/")))) if from_ else None
    to = datetime.date(*(map(int, to.split("/")))) if to else None

    def fn(tweet: Tweet) -> bool:
        if from_ is not None and tweet.date < from_:
            return True
        if to is not None and tweet.date > to:
            return True
        return False

    return fn


def filter_by_keywords(
        keywords: typing.Optional[typing.List[str]] = None,
) -> typing.Callable[[Tweet], bool]:
    """Filter out tweets by keywords.

    Args:
        keywords (typing.List[str]): Keywords used for filtering.

    Returns:
        True if this tweet is to be filtered out. Otherwise, False.
    """
    keywords = keywords if keywords else []
    keywords.extend(["応募", "抽選", "当選"])

    def fn(tweet: Tweet) -> bool:
        return any(keyword in tweet.text for keyword in keywords)

    return fn


def filter_by_langs(
        langs: typing.Optional[typing.List[str]] = None,
) -> typing.Callable[[Tweet], bool]:
    """Filter out tweets by languages. By default, tweets that are not written in Japanese are filtered out.

    Args:
        langs (typing.List[str]): Valid languages.

    Returns:
        True if this tweet is to be filtered out. Otherwise, False.
    """
    langs = langs if langs else ["ja"]

    def fn(tweet: Tweet) -> bool:
        return tweet.lang not in langs

    return fn


# リンクに類する物を除去
def filter_by_links() -> typing.Callable[[Tweet], bool]:
    def fn(tweet: Tweet) -> bool:
        match = re.search(r'ttps?://[\w/:%#\$&\?\(\)~\.=\+\-]+', tweet.raw_text)
        return any([tweet.links != [], match])

    return fn


# メンションを含むツイートを除去
def filter_by_mentions() -> typing.Callable[[Tweet], bool]:
    def fn(tweet: Tweet) -> bool:
        return tweet.mentions != []

    return fn


# 引用リツイートを除去
def filter_by_quote() -> typing.Callable[[Tweet], bool]:
    def fn(tweet: Tweet) -> bool:
        return tweet._is_quote is True

    return fn


# 絵文字を含むツイートを除去
def filter_by_emojis() -> typing.Callable[[Tweet], bool]:
    def fn(tweet: Tweet):
        return any([char in emoji.UNICODE_EMOJI['en'] for char in tweet.raw_text])

    return fn


# length_th以下の文字数のツイートを除去
def filter_by_length(length_th: typing.Optional[int] = None) -> typing.Callable[[Tweet], bool]:
    length_th = length_th if length_th else 20

    def fn(tweet: Tweet):
        return len(tweet.raw_text) <= length_th

    return fn


def collate_fn(features: typing.List[dict]):
    """ミニバッチを作成する関数．

    NOTE: BERT の入力 (input_ids, attention_mask, token_type_ids) は torch.Tensor に変換．それ以外は Python のリストに変換．
    """
    batch = {}
    first = features[0]
    for field in first:
        if field in {"input_ids", "attention_mask", "token_type_ids"}:
            batch[field] = torch.tensor([feature[field] for feature in features])
        else:
            batch[field] = [feature[field] for feature in features]
    return batch
