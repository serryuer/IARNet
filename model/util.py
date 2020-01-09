import torch
import numpy as np


def load_parallel_save_model_to_normal_model(checkpoint, model):
    state_dict = torch.load(checkpoint).state_dict()
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def load_parallel_save_model(checkpoint, model):
    state_dict = torch.load(checkpoint).state_dict()
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    model.load_state_dict(state_dict)
    return model


def read_vectors(path):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vector = np.asarray([float(x) for x in tokens[1:]])

            if first_line:
                first_line = False
                dim = vector.shape[0]
            if vector.shape[0] != dim:
                continue
            vectors[tokens[0]] = vector
            iw.append(tokens[0])
    for i, w in enumerate(iw):
        wi[w] = i
    vectors['UNK'] = np.random.randn(dim)
    return vectors, iw, wi, dim


if __name__ == '__main__':
    vectors, iw, wi, dim = read_vectors('/sdd/yujunshuai/model/en_glove_vector/glove.42B.300d.txt')
    embed = torch.nn.Embedding(195203, 300)
    weight = torch.from_numpy(np.stack(list(vectors.values())))
    torch.save(weight, 'pretrained_weight.pt')
    torch.save(wi, 'vocab.pt')
    embed.weight.data.copy_(weight)
    print(embed)
