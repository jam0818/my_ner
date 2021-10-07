import json

from tqdm import tqdm


def json2dict(path, max_seq_len):
    sents, labels = [], []

    with open(path) as f:
        for idx, line in enumerate(tqdm(f)):
            input_dict = json.loads(line)
            sent = input_dict['words']
            label = input_dict['ner']
            sents.append(sent)
            while len(label) < max_seq_len:
                label.append('O')
            labels.append(label)
            assert len(label) == max_seq_len
    return sents, labels


def main():
    path = '/home/nkawamura/python_project/my_ner/data/samples.jsonl'
    sents, labels = json2dict(path, 128)
    with open('/home/nkawamura/python_project/my_ner/data/samples_pad.jsonl', 'w') as f:
        for sent, label in zip(sents, labels):
            tmp_dict = {'words': sent, 'ner': label}
            tmp = json.dumps(tmp_dict, ensure_ascii=False)
            f.write(tmp)
            f.write('\n')


if __name__ == '__main__':
    main()
