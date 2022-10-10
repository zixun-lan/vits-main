# """ from https://github.com/keithito/tacotron """
#
# '''
# Defines the set of symbols used in text input to the model.
# '''
# _pad        = '_'
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
#
#
# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
#
# # Special symbol ids
# SPACE_ID = symbols.index(" ")

# """ from https://github.com/keithito/tacotron """
#
# '''
# Defines the set of symbols used in text input to the model.
# '''
# _pad        = '_'
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
#
# _punctuation_zh = '；：，。！？-“”《》、 '
#
# _numbers = '1234506789'
#
# # Export all symbols:
# # symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
# symbols = [_pad] + list(_punctuation_zh) +  list(_letters) + list(_numbers) #zhongwen
#
#
#
#
#
#
# # Special symbol ids
# SPACE_ID = symbols.index(" ")
""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
import numpy as np

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


_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

_punctuation_zh = '；：，。！？-“”《》、 '

_numbers = '1234506789'

# Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
old_symbols = [_pad] + list(_punctuation_zh) + list(_letters) + list(_numbers)  # zhongwen

# print(symbols)
cuttor = IMM()
symbols = cuttor.idx2word

# print(symbols)

# Special symbol ids
SPACE_ID = symbols.index(" ")


# print(SPACE_ID)

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
