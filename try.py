# %matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
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
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
from scipy.io.wavfile import write




def chinese_cleaners1(text):
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return ' '.join(phones)


pinyin2pinlv_dict = np.load(r'D:\project\try\pinyin2pinlv\new_pinyin2pinlv_dict.npy', allow_pickle=True).item()


def to_pinlv(cn_pinying):
    # print(cn_pinying.split())
    cn_pinlv = []
    for i in cn_pinying.split():
        if i in pinyin2pinlv_dict.keys():
            cn_pinlv.append(pinyin2pinlv_dict[i])
        else:
            cn_pinlv.append(i)
    # print(cn_pinlv)
    # print(' '.join(cn_pinlv))
    cn_pinlv = ' '.join(cn_pinlv)
    return cn_pinlv

def txt2seq(text):
    cn_pinying = chinese_cleaners1(text)
    clean_text = to_pinlv(cn_pinying)

    sequence = []

    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def get_text(text, hps):
    # text_norm = text_to_sequence(text, hps.data.text_cleaners)
    text_norm = txt2seq(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    print(text_norm)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./logs/ljs_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/G_11000.pth", net_g, None)

# stn_tst = get_text("VITS is Awesome!", hps)
stn_tst = get_text("我是中国人，微软小冰在苏州，我在做变声器项目。", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
print(audio)
# audio.export('./0.wav', format='wav')
print(type(audio))
print(audio.shape)
print(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
aa = ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
print(dir(aa))




















