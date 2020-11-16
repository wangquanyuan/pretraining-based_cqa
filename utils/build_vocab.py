from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(dic, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word, count in dic:
            f.write("%s\t%d\n" % (word, count))


def read_data(path_1, path_2):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2:
        words = []
        for line in f1:
            line = line.strip()

        for line in f2:
            words += line.split()
    return words


def build_vocab(items):
    """
    构建词典
    :param items: list  [item1, item2, ... ]
    :return: list: (word,count)
    """
    dic = defaultdict(int)
    for item in items:
        for i in item.split(" "):
            i = i.strip()
            if not i:
                continue
            dic[i] += 1
    # sort by count
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return dic


if __name__ == '__main__':
    lines = read_data('{}/datas/train_set.csv'.format(BASE_DIR),
                      '{}/datas/dev_set.csv'.format(BASE_DIR))
    vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/datas/vocab.txt'.format(BASE_DIR))