from rsiattack.run import rsi_runner
import os
import torch
import sys
import random
import numpy as np
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

attacks_method = [
    #"l2t",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    #"aitl",
    "htc"
]

model_types = [
    #"vgg16",
    "resnet34",
    #"densenet121",
    #"inception_resv2",
    #"inception_v3",
    #"googlenet",
]

if __name__ == "__main__":
    for model in model_types:
        print("使用源模型为：",model)
        for method in attacks_method:
            print("现在执行攻击方法：",method)
            rsi_runner(method,model)
    