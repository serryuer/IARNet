import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.models import VGG
from transformers import BertModel
import torchvision


class PureVGG(nn.Module):
    def __init__(self, num_class, vgg_trainable=True):
        super(PureVGG, self).__init__()
        self.num_class = num_class
        self.vgg_trainable = vgg_trainable
        self.vgg = torchvision.models.vgg19_bn(pretrained=True).features
        self.img_lin = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 32),
        )

        self.classifier = nn.Linear(32, num_class)

    def _init_parameters_(self):
        torch.nn.init.xavier_uniform(self.img_lin[0].weight)
        torch.nn.init.xavier_uniform(self.img_lin[3].weight)

    def forward(self, input):
        input_ids, attention_masks, img_tensors = input[0], input[1], input[2]
        if self.vgg_trainable:
            img_features = self.vgg(img_tensors)
        else:
            with torch.no_grad():
                img_features = self.vgg(img_tensors)
        img_features =  self.img_lin(img_features.view(img_features.size(0), -1))

        logits = self.classifier(img_features)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), input[3].view(-1))
        outputs = (loss, logits)
        return outputs
