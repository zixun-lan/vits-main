# %matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
from text.symbols import symbols, IMM, to_pinlv_list


cutor = IMM()
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
from scipy.io.wavfile import write




def chinese_cleaners1(text):
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return ' '.join(phones)


pinyin2pinlv_dict = np.load(r'D:\project\try\pinyin2pinlv\pinyin2pinlv_dict.npy', allow_pickle=True).item()


def txt2seq(text):
    cn_pinying = chinese_cleaners1(text)
    print(cn_pinying)
    clean_text = to_pinlv_list(cn_pinying)
    print(clean_text)
    sequence = [_symbol_to_id[symbol] for symbol in clean_text]
    return sequence


def get_text(text, hps):
    # text_norm = text_to_sequence(text, hps.data.text_cleaners)
    text_norm = txt2seq(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    print(text_norm)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./logs/a100_pinlv_token/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cpu()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/a100_pinlv_token/G_142000.pth", net_g, None)




# stn_tst = get_text("我是中国人，微软小冰在苏州，我在做变声器项目。", hps)
stn_tst = get_text('而且口红的外形上还有流苏，它也是爱情的一个象征，是恋人之间的一种信物。这种感觉把流苏赠送给心爱的人，就代表了两个人相守到老的决心。', hps)
#stn_tst = get_text('送完，爱人还可以送姐妹直播间，还有没有想送礼的？在评论区打出来让主播看看还有谁没有送礼啊。', hps)
#stn_tst = get_text('如果有一天，我能够对我们的公共利益有所贡献，我就会认为自己是世界上最幸福的人了。', hps)
with torch.no_grad():
    x_tst = stn_tst.cpu().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    #audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.2, length_scale=1.2)[0][0,0].data.cpu().float().numpy()
print(audio)
# audio.export('./0.wav', format='wav')
print(type(audio))
print(audio.shape)
print(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
from scipy.io.wavfile import write
scaled = np.int32(audio * 2147483648)  #audio是numpy -1到1之间。
write('test.wav', 22050, scaled)


