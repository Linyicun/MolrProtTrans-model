import torch
import torch.nn as nn
import math
import numpy as np
#默认求的是平均值、

input = torch.randn(10, 1)#预测值
target = torch.randn(10,1)#真值
print('input is', input)
print('target is', target)

loss = nn.MSELoss()
output = loss(input, target)
print('output is', output)