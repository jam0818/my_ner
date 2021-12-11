from pathlib import Path
import logging

import torch
from utils import set_device
from tqdm import tqdm

from dataset import MyDataset, MyDataLoader, MyDataset2heads
from argparse import ArgumentParser
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from model import MyBertForTokenClassification, BertForTokenClassification2Heads
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
                        default="/mnt/elm/kawamura/my_NICT_BERT",
                        help='pretrained BERT model path')
    parser.add_argument('--save-path',
                        default='/mnt/elm/kawamura/fine_tuned/twitter-analysis/1022/',
                        help='path to save models')
    parser.add_argument('--train_size',
                        default=4000,
                        type=int,
                        help='path to save models')
    parser.add_argument('--loss_type',
                        default='total',
                        type=str,
                        help='path to save models')
    parser.add_argument('--tag_type',
                        default='BIO',
                        type=str,
                        help='path to save models')

    args = parser.parse_args()
    save_path = Path(args.save_path)
    device = set_device(args.gpuid)
    pretrained_model = args.pretrained_model
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    logger.debug("Create a data loader.")
    logger.debug(f"Load data from {args.dataset_path}.")
    dataset = Path(args.dataset_path)
    train_data_loader = MyDataLoader(dataset / f'train_{args.train_size}_{args.tag_type}.json',
                                     tokenizer,
                                     MyDataset2heads,
                                     special_tokens=['[政府]', '[大衆]'],
                                     batch_size=args.batch_size,
                                     tag_type=args.tag_type)
    dev_data_loader = MyDataLoader(dataset / f'dev_200_{args.tag_type}.json',
                                   tokenizer,
                                   MyDataset2heads,
                                   special_tokens=['[政府]', '[大衆]'],
                                   batch_size=args.batch_size,
                                   tag_type=args.tag_type)

    logger.debug("define a model.")
    model = BertForTokenClassification2Heads(pretrained_model,
                                             num_labels=len(train_data_loader.dataset.id2label['topic'])).to(device)
    optimizer = AdamW(filter(lambda x: x.requires_grad,
                             model.parameters()),
                      lr=args.lr,
                      weight_decay=args.weight_decay)
    num_training_steps = len(train_data_loader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    logger.debug(f"batch size is {args.batch_size}.")
    best_score = -1
    for epoch in range(args.num_epochs):
        model.train()
        model = train(model, train_data_loader, optimizer, scheduler, device, args.loss_type)

        model.eval()
        outputs = evaluate(model, dev_data_loader, device)
        print(f"topic f1 score: {outputs['score']['topic']}")
        print(f"target f1 score: {outputs['score']['target']}")
        total_score = outputs['score']['topic'] + outputs['score']['target']
        torch.save(model.state_dict(),
                   save_path / f'checkpoint_{epoch + 1}_bs{args.batch_size}_ep{args.num_epochs}_{args.train_size}_{args.loss_type}.pth')
        if total_score > best_score:
            torch.save(model.state_dict(),
                       save_path / f'checkpoint_best_bs{args.batch_size}_ep{args.num_epochs}_{args.train_size}_{args.loss_type}.pth')
            best_score = total_score


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="DEBUG")
    main()
