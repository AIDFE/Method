import torch
import torch.nn as nn
import random
import numpy as np
keys = torch.rand(size=(2, 320, 32, 32))
labels = torch.randint(low=0, high=20, size=(2,512,512))
unre_mask = torch.randint(low=0, high=2, size=(2,32,32))
# print(unre_mask)


batch_size = keys.shape[0]
feat_dim = keys.shape[1]
# labels = labels[:, ::16, ::16]
# labels.masked_fill_(unre_mask, 0)
# # print(labels)


this_feat = keys[0].contiguous().view(feat_dim, -1)
print(this_feat.shape)
