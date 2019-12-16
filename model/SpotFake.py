import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.models import VGG
from transformers import BertModel
import torchvision


def load_vgg_from_file(vgg_Path):
    cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    model = VGG(make_layers(cfgs['E'], batch_norm=False))
    state_dict = torch.load(vgg_Path, map_location=None)
    model.load_state_dict(state_dict)
    return model


class SpotFake(nn.Module):
    def __init__(self, num_class, bert_path):
        super(SpotFake, self).__init__()
        self.num_class = num_class
        self.bert = BertModel.from_pretrained(bert_path)
        self.text_lin = nn.Linear(768, 32)

        # self.vgg = load_vgg_from_file(vgg_path).features
        self.vgg = torchvision.models.vgg19_bn(pretrained=True).features
        self.img_lin = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 32),
        )

        self.classifier = nn.Linear(64, num_class)

    def _init_parameters_(self):
        torch.nn.init.xavier_uniform(self.text_lin.weight)
        torch.nn.init.xavier_uniform(self.img_lin[0].weight)
        torch.nn.init.xavier_uniform(self.img_lin[3].weight)

    def forward(self, input):
        input_ids, attention_masks, img_tensors = input[0], input[1], input[2]
        text_features = self.bert(input_ids, attention_masks)[1]
        img_features = self.vgg(img_tensors)
        img_features = img_features.view(img_features.size(0), -1)
        all_features = torch.cat([self.text_lin(text_features), self.img_lin(img_features)], dim=-1)

        logits = self.classifier(all_features)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), input[3].view(-1))
        outputs = (loss, logits)
        return outputs
