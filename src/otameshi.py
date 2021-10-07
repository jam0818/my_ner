from torch.cuda import set_device
from tqdm import tqdm

from dataset import MyDataset, MyDataLoader
from argparse import ArgumentParser
from transformers import BertTokenizer
from model import MyBertForTokenClassification


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/nkawamura/python_project/my_ner/data/samples_pad.jsonl')
    # parser.add_argument('--pretrained_model',
    #                     default="/home/nkawamura/models/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
    #                     help='pretrained BERT model path')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tmp_data_loader = MyDataLoader(args.data_path, tokenizer)
    tmp_bar = tqdm(tmp_data_loader)
    model = MyBertForTokenClassification()
    for batch_idx, batch in enumerate(tmp_bar):
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
    # print(dataset.sents)
    # print(dataset.labels)
    # print(dataset[0]['label'])
        print(output)


if __name__ == '__main__':
    main()
