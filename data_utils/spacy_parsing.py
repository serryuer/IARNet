import spacy

from nltk import DependencyGraph


class SpacyParsing(object):
    def __init__(self):
        self.nlp = spacy.load('en')

    def parse(self, sent, start_index):
        nlp = spacy.load('en')
        doc = nlp(sent)
        par_result = ''
        for token in doc:
            if token.dep_ == 'ROOT':
                head = 0
            else:
                head = token.head.i
            par_result += "\t" + token.text + "(" + token.tag_ + ")" + "\t" + token.tag_ + "\t" + str(
                head) + "\t" + token.dep_ + "\n"
        # print(par_result)
        conlltree = DependencyGraph(par_result)  # 转换为依存句法图
        tree = conlltree.tree()  # 构建树结构

        return tree


if __name__ == '__main__':
    pass
