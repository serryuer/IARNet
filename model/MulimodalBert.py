import torch

from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertEncoder, BertPooler, BertLayer, BertEmbeddings, BertModel

model_urls = {
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class ModiifyResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=True):
        super(ModiifyResNet, self).__init__(block, layers, num_classes=1000, zero_init_residual=False,
                                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                            norm_layer=None)

        state_dict = load_state_dict_from_url(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                                              progress=True)
        self.load_state_dict(state_dict)
        self.linear = nn.Linear(2048, 768)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.transpose(dim0=1, dim1=2)
        x = self.linear(x)

        return x


class MultimodalBert(BertPreTrainedModel):
    def __init__(self, config):
        super(MultimodalBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)

        self.img_encoder = ModiifyResNet(Bottleneck, [3, 8, 36, 3])
        self.sen_encoder = BertLayer(config)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, image_tensor, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):

        input_shape = input_ids.size()
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_extended_attention_mask = None

        head_mask = [None] * self.config.num_hidden_layers

        text_embeddings = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                          token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        text_encoding = self.sen_encoder(text_embeddings,
                                         attention_mask=extended_attention_mask)[0]
        img_encoding = self.img_encoder(image_tensor)

        embedding_output = torch.cat([img_encoding, text_encoding], dim=-2)

        img_mask = torch.ones([attention_mask.shape[0], 49], device=attention_mask.device, dtype=torch.long)
        attention_mask = torch.cat([img_mask, attention_mask], dim=-1)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


class MultimodalBertForClassification(nn.Module):
    def __init__(self, num_class, bert_path):
        super(MultimodalBertForClassification, self).__init__()
        self.num_class = num_class
        self.multiModelBert = MultimodalBert.from_pretrained(bert_path)
        # self.multiModelBert = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(768, num_class)

    def forward(self, input):
        input_ids, attention_mask, image_tensor, labels = input[0], input[1], input[2], input[3]
        # outputs = self.multiModelBert(input_ids, attention_mask)[1]
        outputs = self.multiModelBert(input_ids, attention_mask, image_tensor)
        logits = self.classifier(outputs)
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


if __name__ == '__main__':
    model = MultimodalBertForClassification(num_class=2,
                                            bert_path='/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16')
    model = model.cuda()

    from PIL import Image
    from torchvision import transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    image_tensor = Image.open('../10012t.jpg').convert('RGB')
    image_tensor = transforms(image_tensor)
    image_tensor = image_tensor.cuda()

    input_ids = torch.randint(1, 100, [1, 128])
    attention_mask = torch.cat([torch.ones([1, 100]), torch.zeros([1, 28])], dim=-1)
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    label = torch.tensor([1], device='cuda:0')

    print(model([input_ids, attention_mask, image_tensor.unsqueeze(0), label]))
