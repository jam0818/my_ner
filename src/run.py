from pathlib import Path
import logging

import torch
from torch.cuda import set_device
from tqdm import tqdm

from dataset import MyDataset, MyDataLoader
from argparse import ArgumentParser
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from model import MyBertForTokenClassification
from evaluate import evaluate
from train import train

logger = logging.getLogger(__file__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/mnt/berry/home/kawamura/local_repos/covid-19-twitter-analysis/data/dataset/for_my_ner')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpuid',
                        default=0,
                        help='GPU id (supposed to be using only 1 GPU)')
    parser.add_argument('--max-seq-len', default=128, help='max sequence length for BERT input')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=2e-5, help='learning rate')
    parser.add_argument('--weight-decay',
                        default=0.01,
                        help="penalty to prevent the model weights from having too large values, to avoid overfitting")
    parser.add_argument('--num-epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--warmup-proportion', default=0.033, help="# of warmup steps / # of training steps")
    parser.add_argument('--pretrained-model',
                        default="/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                        help='pretrained BERT model path')
    parser.add_argument('--save-path',
                        default='/mnt/elm/kawamura/fine_tuned/twitter-analysis/1010/',
                        help='path to save models')

    args = parser.parse_args()
    save_path = Path(args.save_path)
    device = set_device(args.gpuid)
    pretrained_model = args.pretrained_model
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    logger.debug("Create a data loader.")
    dataset = Path(args.dataset_path)
    train_data_loader = MyDataLoader(dataset / 'train_topic_1000.json', tokenizer, batch_size=args.batch_size)
    dev_data_loader = MyDataLoader(dataset / 'dev_topic_1000.json', tokenizer, batch_size=args.batch_size)

    logger.debug("define a model.")
    model = MyBertForTokenClassification(pretrained_model)
    optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_data_loader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    logger.debug(f"batch size is {args.batch_size}.")
    best_score = -1
    for epoch in range(args.num_epochs):
        model.train()
        model = train(model, train_data_loader, optimizer, scheduler, device)

        model.eval()
        score = evaluate(model, dev_data_loader, device)
        print(f'f1 score: {score}')
        torch.save(model.state_dict(), save_path / f'checkpoint_{epoch + 1}.pth')
        if score > best_score:
            torch.save(model.state_dict(), save_path / 'checkpoint_best.pth')
            best_score = score


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="DEBUG")
    main()
