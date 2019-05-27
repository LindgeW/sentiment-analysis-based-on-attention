import sys
sys.path.append(["../../", "../", "./"])
import numpy as np
import os
import re
import html
import jieba
import jieba.analyse
from snownlp import SnowNLP

USER_DIR = 'data'  # 用户自定义文件目录
USER_DICT_PATH = os.path.join(USER_DIR, 'user_dict.txt') #自定义用户表
STOP_WORDS_PATH = os.path.join(USER_DIR, 'stop_words.txt') #停用词表
jieba.load_userdict(USER_DICT_PATH)
# jieba.analyse.set_stop_words(STOP_WORDS_PATH)
# pos_tags = ['n', 'vn', 'v', 'ad', 'a', 'e', 'y'] #是名词、形容词、动词、副词、叹词、语气词


# 自定义文本处理工具类
class TextUtils:
    # 加载停用词表
    @classmethod
    def stop_words(cls, stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stp_wds = f.read()  # 读取文件中所有内容
            stp_wdlst = stp_wds.splitlines()  # 按照行分隔，返回一个包含各行作为元素的列表，默认不包含换行符
            return stp_wdlst

    @classmethod
    def is_blank(cls, _str):
        if _str in ['', np.nan, None]:
            return True
        else:
            return False

    @classmethod
    def remove_blank(cls, _str):  # 去除空白字符
        return re.sub(r'\s', '', _str)

    @classmethod
    def html_unescape(cls, _str):  # 将字符串中的html实体转换成html标签
        return html.unescape(_str)

    @classmethod
    def simplified(cls, _str):  # 繁体中文转简体中文
        return SnowNLP(_str).han

    @classmethod
    def maketrans(cls, _str, src, target):  # 将源串中需要转换的字符（串）转换成目标字符（串）
        if src in _str and len(src) == len(target):
            trans_table = str.maketrans(src, target)  # 如：贼贵 -> 很贵
            return _str.translate(trans_table)
        else:
            return _str

    @classmethod
    def tokenize(cls, _str):  # 对字符串列表进行分词
        return jieba.lcut(_str)
        # return [wd for wd in jieba.lcut(_str) if wd not in cls.stop_words(STOP_WORDS_PATH)]

    @classmethod
    def normalize(cls, _str):  # 将数字和字母进行归一化，替换成统一字符 #
        pattern = re.compile(r'[\w]+', re.A | re.M)
        _str = re.sub(pattern, '#', _str)
        return _str

    @classmethod
    def del_repeat_elem_from_list(cls, _list):  # 删除列表中相邻重复元素，如：[1,2,2,3,3,3,4,4,5]->[1,2,3,4,5]
        result = []
        for item in _list:
            size = len(result)
            if size == 0 or result[size - 1] != item:
                result.append(item)
        return result

    @classmethod
    def del_repeat_chars_from_str(cls, _str):  # 删除字符串中连续重复的字符，如：abccdssbb -> abcdsb
        n = len(_str)  # 字符长度
        if n == 1:
            return _str
        _list = list(_str)  # 字符串列表化
        list1 = []
        for i in range(n - 1):
            if _list[i] != _list[i + 1]:
                list1.append(_list[i])
        list1.append(_list[-1])  # 添加末尾字符
        str1 = ''.join(list1)
        return str1

    @classmethod
    def del_repeat_words_from_str(cls, _str):  # 连续重复词语（压缩去重） acabababcdsab -> acabcdsab
        n = len(_str)
        if n <= 1:
            return _str
        rm_list = []
        i = 0
        idx = 0
        while i < n:
            flag = False
            for j in range(n - 1, i, -1):
                if j + j - i <= n:
                    if _str[i: j] == _str[j: (j + j - i)]:
                        rm_list.append([i, j])  # 保存重复序列的前后索引
                        idx = j
                        flag = True
                        break
            if flag:
                i = idx
            else:
                i += 1
        res = _str
        rm_len = 0
        for item in rm_list:
            res = res[:(item[0] - rm_len)] + res[(item[1] - rm_len):]
            rm_len += (item[1] - item[0])
        return res

    @classmethod
    def preprocess(cls, sent):  # 预处理
        sent = cls.remove_blank(sent)
        sent = cls.html_unescape(sent)
        sent = cls.normalize(sent)
        sent = cls.del_repeat_words_from_str(sent)
        sent = cls.simplified(sent)
        if cls.is_blank(sent):
            return ''
        return sent


if __name__ == '__main__':
    x = ['张三 来自湖北 武汉！！！']
    print(TextUtils.preprocess(x))
