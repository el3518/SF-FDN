import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifiers(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifiers, self).__init__()
        self.type = type
        self.class_num = class_num
        self.fcs = nn.Sequential()
        self.fc = {}
        for i in range(class_num):
            if type == 'wn':
                self.fc[i] = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
                self.fc[i].apply(init_weights)
            else:
                self.fc[i] = nn.Linear(bottleneck_dim, class_num)
                self.fc[i].apply(init_weights)
            self.fcs.add_module('fc_'+str(i), self.fc[i])

    def forward(self, x, member=0):
        out=[]
        for i in range(self.class_num):
            outi = self.fcs[i](member[:,i].reshape(x.shape[0],1)*x)
            #xi.type
            out.append(outi)
        #x = self.fc(x)
        return out

class feat_classifierf(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, rule_num=65, type="linear"):
        super(feat_classifierf, self).__init__()
        self.type = type
        self.class_num = class_num
        self.rule_num = rule_num
        self.fcs = nn.Sequential()
        self.fc = {}
        for i in range(rule_num):
            if type == 'wn':
                self.fc[i] = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
                self.fc[i].apply(init_weights)
            else:
                self.fc[i] = nn.Linear(bottleneck_dim, class_num)
                self.fc[i].apply(init_weights)
            self.fcs.add_module('fc_'+str(i), self.fc[i])

    def forward(self, x):
        out=[]
        for i in range(self.rule_num):
            outi = self.fcs[i](x)
            #xi.type
            out.append(outi)
        #x = self.fc(x)
        return out

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

class globe_w(nn.Module):
    def __init__(self, channel, reduction=1):
        super(globe_w, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x)
        return x

'''
class FuzNet2(nn.Module):
    def __init__(self, class_num, base_net = 'resnet50'):
        super(FuzNet2, self).__init__()
        self.netF = network.ResBase(res_name=base_net)
        
        self.netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
        self.netC1 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
        
        netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
        netC2 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
     
        
        
    def forward(self, x):
        x = self.fc(x)
        return x
'''
#########################################################
class generator_fea_deconv(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, input_size=224, class_num=10):
        super(generator_fea_deconv, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.class_num = class_num
        self.batch_size = 64

        # label embedding
        self.label_emb = nn.Embedding(self.class_num, self.input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            # nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.Linear(1024, 128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.mul(self.label_emb(label), input)
        # x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 512, (self.input_size // 32), (self.input_size // 32))
        x = self.deconv(x)
        x = x.view(x.size(0), -1)

        return x

class infoNCE():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num

    def get_posAndneg(self, features, labels, tgt_label=None, feature_q_idx=None, co_fea=None):
        self.features = features
        self.labels = labels

        # get the label of q
        q_label = tgt_label[feature_q_idx]

        # get the positive sample
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = co_fea.unsqueeze(0)


        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.Tensor([]).cuda()
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))


class infoNCE_g():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE_g, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num
        self.fc_infoNCE = nn.Linear(feature_dim, 1).cuda()

    def get_posAndneg(self, features, labels, feature_q_idx=None):
        self.features = features
        self.labels = labels

        # get the label of q
        q_label = self.labels[feature_q_idx]

        # get the positive sample
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label and i != feature_q_idx:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = self.features[feature_q_idx].unsqueeze(0)

        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.tensor([]).cuda()
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))

class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.)*init_weights)   
    
    def forward(self,x):
        x = self.w*torch.ones((x.shape[0]),1).cuda()
        x = torch.sigmoid(x)
        return x
