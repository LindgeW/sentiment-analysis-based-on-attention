import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import config.HyperConfig as Config
import Attention_LSTM as Model
import loss_function as LossFunc
import data
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset


def draw(acc_lst, loss_lst):
	assert len(acc_lst) == len(loss_lst)
	nb_epochs = len(acc_lst)
	plt.subplot(211)
	plt.plot(list(range(nb_epochs)), loss_lst, c='r', label='loss')
	plt.legend(loc='best')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.subplot(212)
	plt.plot(list(range(nb_epochs)), acc_lst, c='b', label='acc')
	plt.legend(loc='best')
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.tight_layout()
	plt.show()


# 测试集评估模型
def evaluate(test_data, classifier, vocab, config):
	# 将本层及子层的training设定为False
	classifier.eval()
	total_acc = 0
	total_loss = 0
	loss_func = nn.CrossEntropyLoss()
	# with torch.no_grad():
	for batch_data in data.get_batch(test_data, config.batch_size):
		# batch_data, seqs_len, _ = data.pad_batch(batch_data, config.max_len)
		corpus_xb, wd2vec_xb, yb, _, seqs_len, _ = data.batch_data_variable(batch_data, vocab)

		out, _ = classifier(corpus_xb, wd2vec_xb, seqs_len)

		total_loss += loss_func(out, yb).item()

		pred = torch.argmax(F.softmax(out, dim=1), dim=1)
		# acc = (pred == yb).sum().item()
		acc = torch.eq(pred, yb).sum().item()
		total_acc += acc

	print('test loss:', total_loss)
	print('test acc:', float(total_acc) / len(test_data))


# 训练模型
def train(train_data, test_data, vocab, config):
	# loss_func = nn.NLLLoss()
	loss_func = nn.CrossEntropyLoss()  # 标签必须为0~n-1，而且必须为1维的
	att_loss = LossFunc.AttentionCrossEntropy()  # 监督注意力损失函数
	embed_weights = vocab.get_embedding_weights(config.embedding_path)

	att_lstm = Model.Attention_LSTM(vocab, config, embed_weights)

	optimizer = Adam(att_lstm.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

	# 3 训练模型
	acc_lst, loss_lst = [], []
	att_lstm.train()  # 将本层及子层的training设定为True
	t1 = time.time()
	for eps in range(config.epochs):
		print(' --Epoch %d' % (1 + eps))
		total_loss = 0
		total_acc = 0
		for batch_data in data.get_batch(train_data, config.batch_size):  # 批训练
			# batch_data, seqs_len, _ = data.pad_batch(batch_data, config.max_len)
			corpus_xb, wd2vec_xb, yb, att_ids, seqs_len, _ = data.batch_data_variable(batch_data, vocab)
			# 3.1 将数据输入模型
			out, weights = att_lstm(corpus_xb, wd2vec_xb, seqs_len)

			# 3.2 重置模型梯度
			att_lstm.zero_grad()
			# optimizer.zero_grad()
			# print(weights.shape, att_ids.shape, out.shape, yb.shape)

			# 3.3 计算误差损失
			loss_cls = loss_func(out, yb)  # 分类误差
			loss_att = att_loss(weights, att_ids)  # 注意力误差
			loss = loss_cls + config.theta * loss_att  # 分类误差+注意力监督误差
			total_loss += loss.item()

			# 计算准确率
			pred = torch.argmax(F.softmax(out, dim=1), dim=1)
			# acc = (pred == yb).sum()
			acc = torch.eq(pred, yb).sum().item()
			total_acc += acc

			# 3.4 反向传播求梯度
			loss.backward()

			# 3.5 (用新的梯度值)更新模型参数
			optimizer.step()

		print('loss:', total_loss)
		print('acc:', float(total_acc) / len(train_data))
		loss_lst.append(total_loss)
		acc_lst.append(float(total_acc) / len(train_data))

	t2 = time.time()
	print('训练总用时：%.2f min' % ((t2-t1) / 60))
	# 绘制acc和loss曲线图
	draw(acc_lst, loss_lst)
	# 保存整个模型
	torch.save(att_lstm, config.save_model_path)
	# 只保存模型参数
	# torch.save(att_lstm.state_dict(), config.save_model_path)

	# 评估模型
	evaluate(test_data, att_lstm, vocab, config)


if __name__ == '__main__':
	np.random.seed(1314)
	torch.manual_seed(3347)
	torch.cuda.manual_seed(3347)
	# torch.backends.cudnn.deterministic = True  # 解决reproducible问题，但是可能会影响性能
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.enabled = False  # cuDNN采用的是不确定性算法，会影响到reproducible

	print('GPU可用：', torch.cuda.is_available())
	print('CuDNN可用：', torch.backends.cudnn.enabled)
	print('GPUs：', torch.cuda.device_count())
	# torch.set_num_threads(4)  # 设定用于并行化CPU操作的OpenMP线程数

	config = Config.Config('config/hyper_param.cfg')

	config.use_cuda = torch.cuda.is_available()

	if config.use_cuda:
		torch.cuda.set_device(0)

	train_data = data.load_data_instance(config.train_data_path)
	test_data = data.load_data_instance(config.test_data_path)

	vocab = data.createVocab(corpus_path=config.train_data_path, lexicon_path=config.lexicon_path)
	vocab.save(config.save_vocab_path)
	train(train_data=train_data, test_data=test_data, vocab=vocab, config=config)
