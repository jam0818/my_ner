import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import filter_by_keywords, filter_by_langs, filter_by_links, \
    filter_by_mentions, filter_by_quote, filter_by_length, filter_by_timestamp, TwitterDataset, collate_fn
from model import BertForTokenClassification2Heads
from utils import set_device, align_predictions, save_df_to_pkl

logger = logging.getLogger(__file__)


def extract(model,
            dataloader,
            device):
    results = {"raw_text": [], "text": [], "sents": [], "timestamp": [], "date": [], "predictions": []}
    with torch.no_grad():
        id2k = dataloader.dataset.id2key
        predictions = {'topic': [], 'target': []}
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # forward
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           token_type_ids=batch['token_type_ids'])
            for idx, key in id2k.items():
                tmp_pred = torch.argmax(output.logits[key], dim=2).tolist()
                tmp_labels = batch['labels'][:, idx].tolist()
                aligned_predictions, _ = align_predictions(predictions=tmp_pred,
                                                           labels=tmp_labels,
                                                           id2label=dataloader.dataset.id2label[key])
                predictions[key].extend(aligned_predictions)
            results["raw_text"].extend(batch["raw_text"])
            results["text"].extend(batch["text"])
            results["sents"].extend(batch["sents"])
            results["timestamp"].extend(batch["timestamp"])
            results["date"].extend(batch["date"])
            results["predictions"].extend(predictions)
        df = pd.DataFrame.from_dict(results)
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/mnt/hinoki/share/covid19/covid-19-twitter-analysis/data/preprocessed'
    )
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
    parser.add_argument('--warmup-proportion', default=0.033, help="# of warmup steps / # of training steps")
    parser.add_argument('--pretrained-model',
                        default="/larch/share/bert/NICT_pretrained_models/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                        help='pretrained BERT model path')
    parser.add_argument('--save-path',
                        default='/mnt/elm/kawamura/topic_target/',
                        help='path to save results')
    parser.add_argument('--filename',
                        default='out.pkl',
                        help='result filename')
    parser.add_argument('--train_size',
                        default=4000,
                        type=int,
                        help='path to save models')
    parser.add_argument('--finetuned-model',
                        type=str,
                        default='/mnt/elm/kawamura/fine_tuned/twitter-analysis/1010/',
                        help='path to finetuned model')
    parser.add_argument('--loss_type',
                        default='total',
                        type=str,
                        help='path to save models')
    parser.add_argument(
        "--start_date",
        default="2021/03/01",
        help="choose start date"
    )
    parser.add_argument(
        "--end_date",
        default="2021/04/30",
        help="choose end date"
    )

    args = parser.parse_args()
    device = set_device(args.gpuid)
    pretrained_model = args.pretrained_model
    logger.debug(f"Create a data loader from {args.dataset_path}.")
    start = args.start_date
    end = args.end_date
    filter_fns = [filter_by_keywords(),
                  filter_by_langs(),
                  filter_by_links(),
                  filter_by_mentions(),
                  filter_by_quote(),
                  filter_by_timestamp(from_=start, to=end),
                  filter_by_length()]
    dataset = TwitterDataset(args.dataset_path,
                             args.model_path,
                             args.max_seq_length,
                             filter_fns)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers)
    model = BertForTokenClassification2Heads(pretrained_model,
                                             num_labels=2).to(device)
    state_dict = torch.load(Path(args.finetuned_model) / f'checkpoint_best_bs{args.batch_size}_ep{args.num_epochs}_{args.train_size}_{args.loss_type}.pth',
                            map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.debug(f"Extract topic and target")
    result_df = extract(model=model,
                        dataloader=dataloader,
                        device=device)
    logger.debug(f"Export results to {args.save_path}")
    output_filename = os.path.join(args.save_path, args.filename)
    save_df_to_pkl(result_df, output_filename)

    if __name__ == '__main__':
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level="DEBUG")
        main()
