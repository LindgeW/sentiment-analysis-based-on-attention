from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def load_data(path):
    wd_lst = []
    wd2vec_lst = []
    i = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            i += 1
            tokens = line.strip().split()
            wd_lst.append(tokens[0])
            wd2vec = np.array(tokens[1:], dtype=np.float32).tolist()
            wd2vec_lst.append(wd2vec)

            if i >= 1000:
                break

    return wd_lst, wd2vec_lst


if __name__ == '__main__':

    wd_lst, wd2vec_lst = load_data('./data/word2vec_100.txt')

    color_map = np.random.randint(0, 3, 1000)
    plt.figure(figsize=(12, 12))
    tsne_data = TSNE(n_components=2, learning_rate=100, init='pca', n_iter=3000).fit_transform(wd2vec_lst)
    print(tsne_data.shape)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], marker='.')
    for i, wd in enumerate(wd_lst):
        # s为注释文本内容
        # xy为被注释的坐标点
        # plt.annotate(s=wd, xy=(tsne_data[i, 0], tsne_data[i, 1]))

        # 设置文字说明：
        # x, y: 表示坐标值
        # s: 表示说明文字
        # fontsize: 表示字体大小
        plt.text(x=tsne_data[i, 0], y=tsne_data[i, 1], s=wd_lst[i], fontsize=10)

    plt.show()