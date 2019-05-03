import torch
import torch.nn as nn
import torch.nn.functional as F


# forward是实际定义损失函数的部分
# 注意：最终返回的必须是一个标量(scalar)，而不是向量(vector)或张量(tensor)
class AttentionCrossEntropy(nn.Module):
	def __init__(self):
		super(AttentionCrossEntropy, self).__init__()

	def forward(self, input, target):  # target为one-hot形式的监督数据
		cross_loss = torch.neg(torch.mul(target.float(), F.log_softmax(input, dim=1)))
		loss = torch.mean(cross_loss)
		return loss
