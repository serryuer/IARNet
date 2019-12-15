# -*- coding: utf-8 -*-

# LTP_DATA_DIR = 'D:/Document/resources/ltp_data_v3.4.0'  # ltp模型目录的路径
LTP_DATA_DIR = '/home/yujunshuai/model/ltp_data_v3.4.0'  # ltp模型目录的路径

import os
from pyltp import Segmentor, Postagger, Parser

from nltk import DependencyGraph


class LtpParsing(object):
    def __init__(self, model_dir=LTP_DATA_DIR):
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(model_dir, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(model_dir, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(model_dir, "parser.model"))

    def parse(self, sent, start_index):
        # 分词
        words = self.segmentor.segment(sent)
        # 词性标注
        postags = self.postagger.postag(words)
        # 句法分析
        arcs = self.parser.parse(words, postags)
        par_result = ''
        for i in range(len(words)):
            if arcs[i].head == 0:
                arcs[i].relation = "ROOT"
            par_result += "\t" + words[i] + \
                          "(" + arcs[i].relation + \
                          "_" + str(start_index) + "-" + str(start_index + len(words[i])) + ")" + \
                          "\t" + postags[i] + "\t" + \
                          str(arcs[i].head) + "\t" + \
                          arcs[i].relation + "\n"
            start_index += len(words[i])
        # print(par_result)
        conlltree = DependencyGraph(par_result)  # 转换为依存句法图
        tree = conlltree.tree()  # 构建树结构
        # tree.draw()  # 显示输出的树
        return tree

    def release_model(self):
        # 释放模型
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()


if __name__ == '__main__':
    ltp = LtpParsing()
    ltp.parse('追回欠款(工程款)的诉讼证据需要哪些呢？', 1)
    ltp.release_model()
