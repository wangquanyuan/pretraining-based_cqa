import pandas as pd
import json
import jieba
from jieba import posseg
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REMOVE_WORDS = [' ', '|', '-']


def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


def parse_data(path1, path2):
    """
    从json中提取处理数据
    :param path1: 读取json文件的路径
    :param path2: 保存处理后数据文件的路径
    """
    file = open(path1, "rb")
    filejson = json.load(file)
    # 提取json中的 paragraphs 部分
    paragraphs = filejson["data"][0]["paragraphs"]
    with open('{}/datas/paragraphs.json'.format(BASE_DIR), 'w') as f:
        json.dump(paragraphs, f)
    data_df = pd.read_json('{}/datas/paragraphs.json'.format(BASE_DIR), encoding='utf-8')
    lines = []
    for row in data_df.itertuples():
        for QA in getattr(row, 'qas'):
            lines.append(getattr(row, 'context')+QA["question"]+"。"+QA["answers"][0]["text"])

    data_df['sen'] = lines
    data = data_df['sen']
    data = data.apply(preprocess_sentence)
    data.to_csv(path2, index=None, header=False, quotechar=' ')#改变quotechar默认值为' '
    print('Save Success ', len(lines))


if __name__ == '__main__':
    parse_data('{}/datas/dureader_robust-data/train.json'.format(BASE_DIR), '{}/datas/train_set.csv'.format(BASE_DIR))
    parse_data('{}/datas/dureader_robust-data/dev.json'.format(BASE_DIR), '{}/datas/dev_set.csv'.format(BASE_DIR))

