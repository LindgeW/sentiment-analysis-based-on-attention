import sys
sys.path.append(["../../", "../", "./"])
from flask import Flask, request, jsonify
import socket
import AttentionModel.config.HyperConfig as Config
from AttentionModel.predict import predict

config = Config.Config('config/hyper_param.cfg')

app = Flask(__name__)


@app.route("/parse", methods=['GET', 'POST'])
def parse():
    print(request.headers)
    if request.method == 'POST':
        req_data = request.json
        print(req_data)
        if not req_data:
            return jsonify({'result': '<strong>no json format！</strong>'})
        elif req_data['text'] is None:
            return jsonify({'result': '<strong>none data！</strong>'})
        else:
            req_lst = req_data['text'].split('|||')
            req_lst = [s.strip() for s in req_lst if s.strip() != '']
            print(req_lst)
            # if len(req_lst) < 2:
            #     return jsonify({'result': '<strong>无效请求数据！</strong>'})

            res_lst = predict(req_lst, config)

            return jsonify({'result': ''.join(res_lst)})
    else:
        return jsonify({'result': '你访问<strong>成功</strong>了！'})


@app.route("/", methods=['GET', 'POST'])
def index():
    X = [
        '鞋子很快就收到了，质量过关，价格不贵，性价比 很高，鞋子穿得也比较舒服，尺码也标准就按平时 自己穿得码数买就行了！',
        '质量很好 版型也很好 码子很标准 穿上很有档次 卖家服务超级好 很满意的一次网上购物',
        '发过来的鞋子跟图片不是同一款 没有图片上的好看 鞋子的鞋面跟鞋带都不一样 只有鞋底一样 太坑了 并且物流不是一般的慢',
        '鞋子感觉一般吧，穿上不是特别舒服，这个价钱中规中矩吧。',
        '感觉一般般，鞋子还是大了，要么就是我的脚变小了',
        '一般了，穿起来感觉舒适度差了点，99不值',
        '鞋子款式简单大方，穿上效果非常不错，物流很快，服务态度好，下次还光顾。好评。',
        '一星给外形设计，快递慢，尺码不标准，偏小。无客服。',
        '#商家，质量有问题也没有人处理，这样的商家怎么不倒闭呢，不要去他家买东西了',
    ]

    res = predict(X, config)
    print(res)
    return '<br><br>'.join(res)


def get_ip():
    return socket.gethostbyname(socket.gethostname())


if __name__ == '__main__':
    ip = get_ip()
    print(ip)
    app.run(ip, port=5000)

