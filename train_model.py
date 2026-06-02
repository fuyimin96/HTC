"""
Description: pre-train model with RSI Dataset
"""

import argparse
import csv
import datetime
import os
import uuid

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import rsiattack
from rsiattack.model import func

import torchvision.transforms as transforms

def resize_tensor_to_299(tensor):
    # tensor: (B, C, H, W)
    return F.interpolate(tensor, size=(299, 299), mode='bilinear', align_corners=False)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="training the models to attack", add_help=add_help
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["UCMerced_LandUse","SIRI_WHU"],
        default="SIRI_WHU",
        help="used trainset in training",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/byf/dataset", help="path of the dataset"
    )
    parser.add_argument("--log_path", type=str, default="/home/byf/Adversarial-Attack/logs/train_logs.csv")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/byf/Adversarial-Attack/checkpoints",
        help="trained model where to save",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def train(args):
    """train backbone models"""

    if args.data_type == "UCMerced_LandUse":
        args.num_classes = 21
    elif args.data_type == "SIRI_WHU":
        args.num_classes = 12

    model_box = func.models_mapping
    
    train_set = rsiattack.trainDataset("train", args)
    test_set = rsiattack.trainDataset("test", args)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=5
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=5
    )

    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    for key, fc in model_box.items():
        net = fc(args.num_classes)  # get net Class from model_box
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        loss = torch.nn.CrossEntropyLoss()
        net = net.to(args.device)
        print(f"** start training {key}, Data is {args.data_type} **")
        for _ in tqdm(range(args.epochs)):
            net.train()
            for data, gt in train_loader:
                data, gt = data.to(args.device), gt.to(args.device)
               # data = resize_tensor_to_299(data)# inc_v3 need input size 299x299
                y_hat = net(data)
                #y_hat = y_hat.logits# for inception_v3
                pred = torch.argmax(y_hat, 1)
                lossrt = loss(y_hat, gt)
                optimizer.zero_grad()
                lossrt.backward()
                optimizer.step()

            scheduler.step()

        """ eval model and print eval result """
        net.eval()
        with torch.no_grad():
            correct_pred = 0
            for data, target in train_loader:
                data, target = data.to(args.device), target.to(args.device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                correct_pred += (pred == target).sum()
            train_acc = correct_pred.item() / len(train_set)

        with torch.no_grad():
            correct_pred = 0
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                correct_pred += (pred == target).sum()
            test_acc = correct_pred.item() / len(test_set)

        if not os.path.exists(args.log_path):
            with open(args.log_path, "w") as f:
                csv_writer = csv.writer(f)
                log_head = [
                    "model_name",
                    "model_type",
                    "train_type",
                    "create_time",
                    "train_acc(%)",
                    "test_acc(%)",
                    "save_path",
                    "backbone",
                ]
                csv_writer.writerow(log_head)
        with open(args.log_path, "a+") as f:
            # write logs in log_path.csv
            random_id = str(uuid.uuid4())
            dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = key + "_" + args.data_type + ".pt"
            save_path = os.path.join(args.save_dir, file_name)
            row_log = [
                file_name,
                key,
                args.data_type,
                dtime,
                f"{train_acc * 100:.4f}",
                f"{test_acc * 100:.4f}",
                save_path,
                str(key),
            ]

            csv_writer = csv.writer(f)
            csv_writer.writerow(row_log)

            torch.save(net.state_dict(), save_path)
        print(
            f"training result is trainacc = {train_acc * 100:04f}%"
            f"testacc = {test_acc * 100:04f}%"
        )
    print("__finish training__")


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args_parser().parse_args()
    train(args)
