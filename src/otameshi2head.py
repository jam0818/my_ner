from tqdm import tqdm

from dataset import MyDataset, MyDataLoader, MyDataset2heads
from argparse import ArgumentParser
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from model import MyBertForTokenClassification
from src.evaluate import evaluate
from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/nkawamura/python_project/my_ner/data/samples_pad.jsonl')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpuid', help='GPU id (supposed to be using only 1 GPU)')
    parser.add_argument('--max-seq-len', default=128, help='max sequence length for BERT input')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', default=2e-5, help='learning rate')
    parser.add_argument('--weight-decay', default=0.01, help="penalty to prevent the model weights from having too large values, to avoid overfitting")
    parser.add_argument('--num-epochs', default=3, help="number of epochs")
    parser.add_argument('--warmup-proportion', default=0.033, help="# of warmup steps / # of training steps")
    # parser.add_argument('--pretrained_model',
    #                     default="/home/nkawamura/models/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
    #                     help='pretrained BERT model path')
    args = parser.parse_args()
    path = '/home/nkawamura/models/bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE'
    tokenizer = BertTokenizer.from_pretrained(path)
    dataset = MyDataset2heads
    train_data_loader = MyDataLoader(args.data_path, tokenizer, dataset)
    model = MyBertForTokenClassification(path, num_labels=5)
    for idx, batch in enumerate(tqdm(train_data_loader)):
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])


if __name__ == '__main__':
    main()
