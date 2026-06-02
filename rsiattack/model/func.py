import os

import torch
import torchvision.models as models

import urllib.request

def vgg16(num):
    try:
        from torchvision.models import VGG16_Weights

        net = models.vgg16(weights=VGG16_Weights.DEFAULT)
    except Exception as exc:
        print(f"[Warn] Failed to load pretrained VGG16 weights: {exc}")
        print("[Warn] Fallback to random initialization (weights=None).")
        net = models.vgg16(weights=None)
    num_ftrs = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(num_ftrs, num)
    return net


def vgg19(num):
    net = models.vgg19()
    num_ftrs = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(num_ftrs, num)
    return net


def resnet34(num):
    net = models.resnet34()
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


def resnet50(num):
    net = models.resnet50()
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


def densenet121(num):
    net = models.densenet121()
    num_ftrs = net.classifier.in_features
    net.classifier = torch.nn.Linear(num_ftrs, num)
    return net


def inception_resv2(num):
    from .inception_resnet_v2 import Inception_ResNetv2

    net = Inception_ResNetv2(classes=num)
    num_ftrs = net.linear.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


def googlenet(num):
    from torchvision.models import GoogLeNet_Weights
    net = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net

def inception_v3(num):
    from torchvision.models import Inception_V3_Weights
    net = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    #net.Conv2d_1a_3x3.conv = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


models_mapping  = {  
    "vgg16": vgg16,
    "resnet34": resnet34,
    "densenet121": densenet121,
    "inception_resv2": inception_resv2,
    "inception_v3": inception_v3,
    "googlenet": googlenet,
}

def get_model_with_pretrain(madel_type, args):
    model_path = os.path.join(args.model_dir, madel_type)
    model_dict = torch.load(model_path, map_location="cpu")
    fc = models_mapping[madel_type]
    net = fc(args.num)
    net.load_state_dict(model_dict)
    return net
