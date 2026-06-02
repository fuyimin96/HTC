import datetime
import os
import sys
import time

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from .data.data_load import attackDataset, evalDataset
from .model.func import models_mapping_simple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
import torch.distributed as dist

# sys.path.append("model")
# sys.path.append(".")



class ATTACK(object):
    def __init__(self, args):
        if args.data_type == "UCMerced_LandUse":
            args.num_classes = 21
        elif args.data_type == "SIRI_WHU":
            args.num_classes = 12

        if not os.path.exists("/home/byf/Adversarial-Attack/logs"):
            os.makedirs("/home/byf/Adversarial-Attack/logs")

        self.white_model_type = args.model_type
        fc = models_mapping_simple[args.model_type]
        model_name = self.args.model_type + "_" + self.args.data_type + ".pt"
        model_path = os.path.join(self.args.mod_dir, model_name)
        model_dict = torch.load(model_path, map_location="cpu",weights_only=True)
        self.net = fc(self.args.num_classes)
        self.net.load_state_dict(model_dict)
        self.net.eval()

        self.check_save_dir()
        self.dataset = attackDataset(args)
        self.label_dict = self.dataset.label_dict
        self.name_dict = self.dataset.name_dict

        #多gpu训练
        if dist.is_available() and dist.is_initialized():
            # 只有在DDP初始化后才使用DistributedSampler
            self.dataset_sampler = DistributedSampler(self.dataset)
        else:
            # 单机模式使用普通sampler
            self.dataset_sampler = RandomSampler(self.dataset)

    def create_options(self, parser):
        parser.add_argument(
            "--data_type", type=str, default="SIRI_WHU", help="used trainset in training"
        )
        #parser.add_argument("--model_type",type=str,default="googlenet",help="used trainset in training",)
        parser.add_argument(
            "--data_dir", type=str, default="/data/byf/dataset", help="path of the dataset"
        )
        parser.add_argument("--img_save_dir", type=str, default="/data/byf/Adversarial-Attack/examples")
        parser.add_argument(
            "--mod_dir",
            type=str,
            default="/home/byf/Adversarial-Attack/checkpoints",
            help="trained model where to save",
        )
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--device", type=str, default="cuda:0")
        return parser

    def eval(self):
        """Evaluate the adversarial images by calculating metrics
            such as success rate and other relevant indicators.

        Args:
            adv_imag (_tensor_): The dataset of attacked images.
        """
        device = self.args.device
        dataset = evalDataset(self.save_new_dir, self.args)
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=5
        )

        df = pd.DataFrame(columns=["fr", "m2c", "c2m"])

        for key, fc in models_mapping_simple.items():
            model_name = key + "_" + self.args.data_type + ".pt"
            model_path = os.path.join(self.args.mod_dir, model_name)
            model_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            net = fc(self.args.num_classes)
            net.load_state_dict(model_dict)
            net.eval()
            net = net.to(device)
            with torch.no_grad():
                fool_rate, all_sum = 0, 0
                for image_clean, image_adv in loader:
                    image_clean = image_clean.to(device)
                    image_adv = image_adv.to(device)
                    pred_clean = torch.argmax(net(image_clean), 1)
                    pred_adv = torch.argmax(net(image_adv), 1)
                    all_sum += image_clean.shape[0]
                    fool_rate += (pred_clean != pred_adv).sum()

                fool_rate = fool_rate / all_sum
            pd.options.display.float_format = "{:.4f}".format
            df.loc[key, "fr"] = fool_rate.item() * 100

            print(f"** eval adv with model {key} **")
            print(f"** dataset is {self.args.data_type} **")
            print(f"fool rate = {fool_rate* 100:2f}")

        print("** finish eval **")
        dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        logs_name = (
            f"/home/byf/Adversarial-Attack/logs/eval_logs/{self.__class__.__name__}"
            f"_{self.args.data_type}_{self.white_model_type}_{dtime}.csv"
        )

        with open(logs_name, "a", encoding="utf-8") as f:
            df.to_csv(f)
            for args_key, args_value in vars(self.args).items():
                f.write(f"{args_key} : {args_value}\n")
        return df

    def check_save_dir(self):
        args = self.args

        path_datatime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        folder_name = f"{args.data_type}_{args.model_type}_{args.att}"

        save_dir = os.path.join(args.img_save_dir, folder_name, path_datatime)

        os.makedirs(save_dir, exist_ok=True)
        self.save_new_dir = save_dir

    def save_images(self, images_adv, labels, files_name):
        for image_adv, label, file_name in zip(images_adv, labels, files_name):
            class_name = self.name_dict[label.item()]
            save_dir = os.path.join(self.save_new_dir, class_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, file_name)
            pil_image = TF.to_pil_image(image_adv)
            pil_image.save(save_path)

    def get_tar_labels(self, images):
        self.net.eval()
        images = images.to(self.args.device)
        with torch.no_grad():
            gt = torch.argmin(self.net(images), 1)
        return gt
