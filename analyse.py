import torch
import torch.nn as nn
from torch.utils import data
from dataloader import PropDataset, pad
from params import params

from collections import Counter

for ias in [params.trainset, params.validset]:
  _dataset = PropDataset(ias, False)

  _iter = data.DataLoader(dataset=_dataset,
                         batch_size=params.batch_size,
                         shuffle=True,
                         num_workers=1,
                         collate_fn=pad)

  print("----------------------")
  tagzz = []
  for xxx in _iter:
    words, x, is_heads, att_mask, tags, y, seqlens = xxx
    tagzz.extend(y[0].view(-1).tolist())

  c = Counter(tagzz)
  print(c)

