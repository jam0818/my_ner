from tqdm import tqdm

from dataset import MyDataset, MyDataLoader, MyDataset2heads
from argparse import ArgumentParser
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from model import MyBertForTokenClassification, BertForTokenClassification2Heads
from src.evaluate import evaluate
from train import train
import json


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/home/nkawamura/python_project/my_ner/data/samples_pad_span.jsonl')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpuid', help='GPU id (supposed to be using only 1 GPU)')
    parser.add_argument('--max-seq-len', default=128, help='max sequence length for BERT input')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', default=2e-5, help='learning rate')
    parser.add_argument('--weight-decay',
                        default=0.01,
                        help="penalty to prevent the model weights from having too large values, to avoid overfitting")
    parser.add_argument('--num-epochs', default=3, help="number of epochs")
    parser.add_argument('--warmup-proportion', default=0.033, help="# of warmup steps / # of training steps")
    # parser.add_argument('--pretrained_model',
    #                     default="/home/nkawamura/models/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
    #                     help='pretrained BERT model path')
    parser.add_argument('--tag_type',
                        default='BIO',
                        type=str,
                        help='path to save models')
    args = parser.parse_args()
    path = '/home/nkawamura/models/bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE'
    tokenizer = BertTokenizer.from_pretrained(path)
    train_data_loader = MyDataLoader(args.data_path,
                                     tokenizer,
                                     MyDataset2heads,
                                     special_tokens=['[政府]', '[大衆]'],
                                     batch_size=args.batch_size,
                                     tag_type=args.tag_type)
    dev_data_loader = MyDataLoader(args.data_path,
                                   tokenizer,
                                   MyDataset2heads,
                                   special_tokens=['[政府]', '[大衆]'],
                                   batch_size=args.batch_size,
                                   tag_type=args.tag_type)
    model = BertForTokenClassification2Heads(path, num_labels=len(train_data_loader.dataset.id2label['topic']))
    optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_data_loader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    for epoch in range(args.num_epochs):
        model.train()
        model = train(model, train_data_loader, optimizer, scheduler, device='cpu', loss_type='total')

        model.eval()
        output_dict = evaluate(model, dev_data_loader, device='cpu')
        with open('/home/nkawamura/python_project/my_ner/data/sample_pred.json', 'w') as f:
            prediction = {}
            for key, pred in output_dict['prediction'].items():
                prediction[key] = pred
            prediction = json.dumps(prediction, ensure_ascii=False)
            f.write(prediction)


if __name__ == '__main__':
    main()
