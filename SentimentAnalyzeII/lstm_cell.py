import torch
import torch.nn as nn

'''
LSTMCell
输入: input, (h_0, c_0)
	input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable
	h_0 (batch, hidden_size): 保存着batch中每个元素的初始化隐状态的Tensor
	c_0 (batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出：h_1, c_1
	h_1 (batch, hidden_size): 下一个时刻的隐状态。
	c_1 (batch, hidden_size): 下一个时刻的细胞状态。


LSTM
输入: input, (h_0, c_0)
	input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])
	h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
	c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出: output, (h_n, c_n)
	output (seq_len, batch, hidden_size * num_directions): 保存RNN最后一层的输出的Tensor。 如果输入是torch.nn.utils.rnn.PackedSequence，那么输出也是torch.nn.utils.rnn.PackedSequence。
	h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
	c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的细胞状态。
'''


class MyLSTM(nn.Module):
	def __init__(self, input_size=0, hidden_size=0, num_layers=1, batch_first=False, bidirectional=False, drop_out=0.0):
		super(MyLSTM, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.bidirectional = bidirectional
		self.drop_out = drop_out
		self.num_directions = 2 if self.bidirectional else 1

		self.fw_cells, self.bw_cells = [], []
		for layer_i in range(self.num_layers):
			layer_input_size = self.input_size if layer_i == 0 else self.num_directions * self.hidden_size
			self.fw_cells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))
			if self.bidirectional:
				self.bw_cells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))

		# self.cell = nn.LSTMCell(
		# 	input_size=self.input_size,   # 输入的特征维度
		# 	hidden_size=self.hidden_size  # 隐层的维度
		# )

	def init_hidden(self, batch_size=1):
		torch.manual_seed(3357)
		h0 = torch.randn(batch_size, self.hidden_size)
		c0 = torch.randn(batch_size, self.hidden_size)
		return h0, c0

	# lstm前向传播
	@staticmethod
	def _forward(lstm_cell, inputs, init_hidden, drop_mask=None):
		assert isinstance(init_hidden, tuple)
		hx_fw = init_hidden
		out_fw = []
		seq_len = inputs.size(0)
		for xi in range(seq_len):
			hi, ci = lstm_cell(inputs[xi], hx_fw)
			out_fw.append(hi)

			if drop_mask is not None:  # 循环层使用dropout
				hi = hi * drop_mask

			hx_fw = (hi, ci)

		out_fw = torch.stack(tuple(out_fw), dim=0)
		return out_fw, hx_fw

	# lstm反向传播
	@staticmethod
	def _backward(lstm_cell, inputs, init_hidden, drop_mask=None):
		assert isinstance(init_hidden, tuple)
		hx_bw = init_hidden
		out_bw = []
		seq_len = inputs.size(0)
		for xi in reversed(range(seq_len)):
			hi, ci = lstm_cell(inputs[xi], hx_bw)
			out_bw.append(hi)

			if drop_mask is not None:  # 循环层使用dropout
				hi = hi * drop_mask

			hx_bw = (hi, ci)

		out_bw.reverse()
		out_bw = torch.stack(tuple(out_bw), dim=0)
		return out_bw, hx_bw

	# 默认inputs: [seq_len, batch_size, input_size]
	# batch_first: [batch_size, seq_len, input_size]
	def forward(self, inputs, init_hidden=None):
		assert torch.is_tensor(inputs) and inputs.dim() == 3

		if self.batch_first:
			inputs = inputs.permute(1, 0, 2)

		batch_size = inputs.size(1)
		if init_hidden is None:
			init_hidden = self.init_hidden(batch_size)

		hx = init_hidden

		hn, cn = [], []
		for layer in range(self.num_layers):
			input_drop_mask, hidden_drop_mask = None, None
			seq_len, batch_size, input_size = inputs.size()
			if self.training:
				print('use dropout...')
				if layer != 0:
					input_drop_mask = torch.empty(batch_size, input_size).fill_(1 - self.drop_out)
					input_drop_mask = torch.bernoulli(input_drop_mask)
					input_drop_mask = torch.div(input_drop_mask, (1 - self.drop_out))
					input_drop_mask = input_drop_mask.unsqueeze(-1).expand((-1, -1, seq_len)).permute((2, 0, 1))
					inputs = inputs * input_drop_mask

				hidden_drop_mask = torch.empty(batch_size, self.hidden_size).fill_(1 - self.drop_out)
				hidden_drop_mask = torch.bernoulli(hidden_drop_mask)     # 以输入值为概率p输出1，(1-p)输出0
				hidden_drop_mask = torch.div(hidden_drop_mask, (1 - self.drop_out))  # 保证训练和预测时期望值一致

			out_fw, (hn_f, cn_f) = MyLSTM._forward(lstm_cell=self.fw_cells[layer], inputs=inputs, init_hidden=hx, drop_mask=hidden_drop_mask)
			# print(out_fw.shape, hn_f.shape, cn_f.shape)

			out_bw, hn_b, cn_b = None, None, None
			if self.bidirectional:
				out_bw, (hn_b, cn_b) = MyLSTM._backward(lstm_cell=self.bw_cells[layer], inputs=inputs, init_hidden=hx, drop_mask=hidden_drop_mask)
			# print(out_bw.shape, hn_b.shape, cn_b.shape)

			hn.append(torch.cat((hn_f, hn_b), dim=1) if self.bidirectional else hn_f)
			cn.append(torch.cat((cn_f, cn_b), dim=1) if self.bidirectional else cn_f)

			inputs = torch.cat((out_fw, out_bw), dim=2) if self.bidirectional else out_fw
			# print('input shape:', inputs.shape)   # (6, 3, 10)

		hn = torch.stack(tuple(hn), dim=0)
		cn = torch.stack(tuple(cn), dim=0)

		output = inputs.permute((1, 0, 2)) if self.batch_first else inputs

		return output, (hn, cn)


if __name__ == '__main__':

	inputs = torch.rand(6, 3, 20)  # [seq_len, batch_size, input_size]
	lstm = MyLSTM(input_size=20, hidden_size=100, num_layers=3, bidirectional=True, drop_out=0.2)
	# h0, c0 = torch.randn(3, 10), torch.randn(3, 10)
	# out, (hn, cn) = lstm(inputs, (h0, c0))
	out, (hn, cn) = lstm(inputs)
	print(out.shape)  # [6, 3, 20]
	print(hn.shape, cn.shape)  # [2, 3, 20]  [2, 3, 20]
