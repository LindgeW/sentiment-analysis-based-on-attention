import time
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import config.HyperConfig as Config
from classifier import SentimentModel
import loss_function as LossFunc
from dataloader import *


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
	for batch_data in get_batch(test_data, config.batch_size):
		corpus_xb, wd2vec_xb, yb, att_ids, mask = batch_data_variable(batch_data, vocab)

		if config.use_cuda:
			corpus_xb = corpus_xb.cuda()
			wd2vec_xb = wd2vec_xb.cuda()
			yb = yb.cuda()
			mask = mask.cuda()

		out, _ = classifier(corpus_xb, wd2vec_xb, mask)
		pred = torch.argmax(out, dim=1)
		acc = torch.eq(pred, yb).cpu().sum().item()
		total_acc += acc

	print('test acc:', float(total_acc) / len(test_data))


# 训练模型
def train(train_data, dev_data, test_data, vocab, config):
	loss_func = nn.CrossEntropyLoss()  # 标签必须为0~n-1，而且必须为1维的
	att_loss = LossFunc.AttentionCrossEntropy()  # 监督注意力损失函数

	embed_weights = vocab.get_embedding_weights(config.embedding_path)
	vocab.save(config.save_vocab_path)

	classifier = SentimentModel(vocab, config, embed_weights)
	optimizer = Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
					 lr=config.learning_rate,
					 weight_decay=config.weight_decay)

	if config.use_cuda:
		classifier = classifier.cuda()

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# # 用.to(device)来决定模型使用GPU还是CPU
	# classifier = classifier.to(device)

	# 3 训练模型
	train_acc_lst, train_loss_lst = [], []
	dev_acc_lst, dev_loss_lst = [], []

	for eps in range(config.epochs):
		classifier.train()  # 将本层及子层的training设定为True
		print(' --Epoch %d' % (1 + eps))
		train_loss, train_acc = 0, 0
		t1 = time.time()
		for batch_data in get_batch(train_data, config.batch_size):  # 批训练
			corpus_xb, wd2vec_xb, yb, att_ids, mask = batch_data_variable(batch_data, vocab)

			if config.use_cuda:
				corpus_xb = corpus_xb.cuda()
				wd2vec_xb = wd2vec_xb.cuda()
				yb = yb.cuda()
				att_ids = att_ids.cuda()
				mask = mask.cuda()

			# corpus_xb = corpus_xb.to(device)
			# wd2vec_xb = wd2vec_xb.to(device)
			# yb = yb.to(device)
			# att_ids = att_ids.to(device)
			# mask = mask.to(device)

			# 3.1 重置模型梯度
			classifier.zero_grad()
			# optimizer.zero_grad()

			# 3.2 将数据输入模型
			out, weights = classifier(corpus_xb, wd2vec_xb, mask)

			# 3.3 计算误差损失
			# loss_cls = loss_func(out, yb)  # 分类误差
			# loss_att = att_loss(weights, att_ids)  # 注意力误差
			# loss = loss_cls + config.theta * loss_att  # 分类误差+注意力监督误差
			loss = loss_func(out, yb)  # 分类误差
			loss_att = att_loss(weights, att_ids)  # 注意力误差
			loss.add_(config.theta * loss_att)  # 分类误差+注意力监督误差
			train_loss += loss.data.cpu().item()

			# 计算准确率
			pred = torch.argmax(out, dim=1)
			acc = torch.eq(pred, yb).cpu().sum().item()
			train_acc += acc

			# 3.4 反向传播求梯度
			loss.backward()

			# 3.5 (用新的梯度值)更新模型参数
			optimizer.step()

		t2 = time.time()
		print('训练用时：%.3f min' % ((t2 - t1) / 60))
		print('train_loss: %.3f  train_acc: %.3f' % (train_loss, float(train_acc) / len(train_data)))
		# print('train_loss: {:.3f}  train_acc: {:.3f}'.format(train_loss, float(train_acc) / len(train_data)))
		train_loss_lst.append(train_loss)
		train_acc_lst.append(float(train_acc) / len(train_data))

		with torch.no_grad():
			classifier.eval()
			dev_acc, dev_loss = 0, 0
			for batch_data in get_batch(dev_data, config.batch_size):
				corpus_xb, wd2vec_xb, yb, att_ids, mask = batch_data_variable(batch_data, vocab)

				if config.use_cuda:
					corpus_xb = corpus_xb.cuda()
					wd2vec_xb = wd2vec_xb.cuda()
					yb = yb.cuda()
					att_ids = att_ids.cuda()
					mask = mask.cuda()

				# corpus_xb = corpus_xb.to(device)
				# wd2vec_xb = wd2vec_xb.to(device)
				# yb = yb.to(device)
				# att_ids = att_ids.to(device)
				# mask = mask.to(device)

				out, weights = classifier(corpus_xb, wd2vec_xb, mask)
				# 3.3 计算误差损失
				# loss_cls = loss_func(out, yb)  # 分类误差
				# loss_att = att_loss(weights, att_ids)  # 注意力误差
				# loss = loss_cls + config.theta * loss_att  # 分类误差+注意力监督误差
				loss = loss_func(out, yb)  # 分类误差
				loss_att = att_loss(weights, att_ids)  # 注意力误差
				loss.add_(config.theta * loss_att)  # 分类误差+注意力监督误差
				dev_loss += loss.data.cpu().item()

				# 计算准确率
				pred = torch.argmax(out, dim=1)
				acc = torch.eq(pred, yb).cpu().sum().item()
				dev_acc += acc

			print('dev_loss: %f  dev_acc: %f' % (dev_loss, float(dev_acc) / len(dev_data)))
			dev_loss_lst.append(dev_loss)
			dev_acc_lst.append(float(dev_acc) / len(dev_data))

	# 绘制acc和loss曲线图
	# draw(train_acc_lst, train_loss_lst)
	# draw(dev_acc_lst, dev_loss_lst)

	# 保存整个模型
	torch.save(classifier, config.save_model_path)
	# 只保存模型参数
	# torch.save(classifier.state_dict(), config.save_model_path)

	# 评估模型
	evaluate(test_data, classifier, vocab, config)


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

	# if torch.cuda.is_available():
	# 	config.device = torch.device("cuda", 0)
	# else:
	# 	config.device = torch.device("cpu")
	# config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_data = load_data_instance(config.train_data_path)
	dev_data = load_data_instance(config.dev_data_path)
	test_data = load_data_instance(config.test_data_path)
	print('train data size:', len(train_data))
	print('dev data size:', len(dev_data))
	print('test data size:', len(test_data))
	vocab = create_vocab(corpus_path=config.train_data_path,
						 lexicon_path=config.lexicon_path)

	train(train_data=train_data, dev_data=dev_data, test_data=test_data, vocab=vocab, config=config)
