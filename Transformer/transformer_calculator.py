import torch
import torch.nn as nn

nums = '0 1 2 3 4 5 6 7 8 9'
n_num = len(nums)
d_hidn = 128
nn_emb = nn.Embedding(n_num, d_hidn)

input_embs = nn_emb(inputs) # input embedding
print(input_embs.size())