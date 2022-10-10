from text.symbols import old_symbols
import numpy as np
from text.symbols import symbols, IMM


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
pinyin2pinlv_dict = np.load('pinyin2pinlv_dict.npy', allow_pickle=True).item()
path = 'pinlv_token_dict.txt'


class IMM(object):
    def __init__(self, dic_path=path):
        self.dictionary = set()
        self.word2idx = {}
        self.idx2word = []
        self.maximum = 0

        # 读取词典
        for line in old_symbols:
            self.dictionary.add(line)
            self.idx2word.append(line)
            self.word2idx[line] = len(self.idx2word) - 1
            if len(line) > self.maximum:
                self.maximum = len(line)

        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.idx2word.append(line)
                self.word2idx[line] = len(self.idx2word) - 1
                if len(line) > self.maximum:
                    self.maximum = len(line)

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index - size):index]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result[::-1]


def chinese_cleaners1(text):
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return ' '.join(phones)


pinyin2pinlv_dict = np.load('pinyin2pinlv_dict.npy', allow_pickle=True).item()


def qu_kong(string):
    string = string.split()
    string = ''.join(string)
    return string


def to_pinlv(cn_pinying):
    # print(cn_pinying.split())
    cn_pinlv = []
    for i in cn_pinying.split():
        if i in pinyin2pinlv_dict.keys():
            cn_pinlv.append(qu_kong(pinyin2pinlv_dict[i]))
        else:
            cn_pinlv.append(i)
    # print(cn_pinlv)
    # print(' '.join(cn_pinlv))
    cn_pinlv = ' '.join(cn_pinlv)
    return cn_pinlv


def to_pinlv_list(cn_pinying):
    # print(cn_pinying.split())
    cn_pinlv_list = []
    for i in cn_pinying.split():
        if i in pinyin2pinlv_dict.keys():
            tmp = pinyin2pinlv_dict[i]
            for j in tmp.split():
                cn_pinlv_list.append(j)
            cn_pinlv_list.append(' ')
        else:
            if i.replace('u', 'v') in pinyin2pinlv_dict.keys():
                tmp = pinyin2pinlv_dict[i.replace('u', 'v')]
                for j in tmp.split():
                    cn_pinlv_list.append(j)
                cn_pinlv_list.append(' ')
            else:
                for ii in i:
                    cn_pinlv_list.append(ii)
                cn_pinlv_list.append(' ')
    cn_pinlv_list = cn_pinlv_list[0:-1]
    return cn_pinlv_list


if __name__ == "__main__":
    text = '咱们用完平衡粉底液以后，不要以为化妆就结束了。爱意满满。全要。'
    print(text)
    cn_pinyin = chinese_cleaners1(text)
    print(cn_pinyin)
    cn_pl_list = to_pinlv_list(cn_pinyin)
    print(cn_pl_list)
    sequence = [_symbol_to_id[symbol] for symbol in cn_pl_list]
    print(sequence)


    # cn_pinlv = to_pinlv(cn_pinyin)
    # print(cn_pinyin)
    # print(cn_pinlv)
    # cut_token = IMM()
    # print(cut_token.cut(cn_pinlv))
    #
    # aaa = [cut_token.word2idx[i] for i in cut_token.cut(cn_pinlv)]
    # print(aaa)
    # print(len(cut_token.word2idx))
    # print(cut_token.word2idx)

    # print(pinyin2pinlv_dict)
    # aa = set()
    # for i in pinyin2pinlv_dict.items():
    #     print(i[1].split())
    #     for j in i[1].split():
    #         print(j)
    #         aa.add(j)
    # print(len(pinyin2pinlv_dict) * 3)
    # print(symbols)

    # print(aa)
    # for i in list(aa):
    #     if i in symbols:
    #         print(111111)
    #     else:
    #         fff = open('pinlv_token_dict', 'a', encoding='utf-8')
    #         fff.write(i)
    #         fff.write('\n')
