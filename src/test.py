import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BertTokenizer

from dataset import MyDataLoader, MyDataset2heads
from evaluate import evaluate
from model import BertForTokenClassification2Heads
from utils import set_device

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
                        default='/mnt/elm/kawamura/fine_tuned/twitter-analysis/BIO/',
                        help='path to save models')
    parser.add_argument('--train_size',
                        default=4000,
                        type=int,
                        help='path to save models')
    parser.add_argument('--finetuned-model',
                        type=str,
                        default='/mnt/elm/kawamura/fine_tuned/twitter-analysis/BIO/',
                        help='path to finetuned model')
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

    logger.debug(f"Create a data loader from {args.dataset_path}.")
    dataset = Path(args.dataset_path)
    logger.debug(f"batch size is {args.batch_size}.")
    test_data_loader = MyDataLoader(dataset / f'test_200_{args.tag_type}.json',
                                    tokenizer,
                                    MyDataset2heads,
                                    special_tokens=['[政府]', '[大衆]'],
                                    batch_size=args.batch_size,
                                    tag_type=args.tag_type)

    logger.debug("define a model.")
    logger.debug(f"load from {args.finetuned_model}")
    model = BertForTokenClassification2Heads(pretrained_model,
                                             num_labels=len(test_data_loader.dataset.id2label['topic'])).to(device)
    state_dict = torch.load(Path(args.finetuned_model) / f'checkpoint_best_bs{args.batch_size}_ep{args.num_epochs}_{args.train_size}_{args.loss_type}.pth',
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    output_dict = evaluate(model, test_data_loader, device)
    print(f"topic f1 score: {output_dict['score']['topic']}")
    print(f"target f1 score: {output_dict['score']['target']}")
    prediction = {}
    labels = {}
    for key, pred in output_dict['prediction'].items():
        prediction[key] = pred
    for key, label in output_dict['labels'].items():
        labels[key] = label
    prediction = json.dumps(prediction, ensure_ascii=False)
    labels = json.dumps(labels, ensure_ascii=False)
    logger.debug(f"save file at {save_path}")
    with open(save_path / f'prediction_bs{args.batch_size}_ep{args.num_epochs}_{args.loss_type}.json', 'w') as f:
        f.write(prediction)
    with open(save_path / f'labels_bs{args.batch_size}_ep{args.num_epochs}_{args.loss_type}.json', 'w') as f:
        f.write(labels)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="DEBUG")
    main()
