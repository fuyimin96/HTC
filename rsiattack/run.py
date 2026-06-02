import argparse

import rsiattack

attacks_box = {
    "di2fgsm": rsiattack.DI2_FGSM,
    "mifgsm": rsiattack.MI_FGSM,
    "l2t":rsiattack.L2T,
    "htc":rsiattack.HTC,
    "aitl":rsiattack.AITL,
}

def rsi_runner(method,model):
    parser = argparse.ArgumentParser(description="training the models to attack")
    parser.add_argument("--method", type=str, default=method)
    parser.add_argument("--model_type", type=str, default=model, help="used trainset in training",)
    args = parser.parse_args()
    attack_fun = attacks_box[args.method]
    attack = attack_fun(parser)
    attack.attack()
