import glob
import os

import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from ..model import func


model_box = func.models_mapping

label_UCMerced_LandUse = {
    "agricultural": 0,
    "airplane": 1,
    "baseballdiamond": 2,
    "beach": 3,
    "buildings": 4,
    "chaparral": 5,
    "denseresidential": 6,
    "forest": 7,
    "freeway": 8,
    "golfcourse": 9,
    "harbor": 10,
    "intersection": 11,
    "mediumresidential": 12,
    "mobilehomepark": 13,
    "overpass": 14,
    "parkinglot": 15,
    "river": 16,
    "runway": 17,
    "sparseresidential": 18,
    "storagetanks": 19,
    "tenniscourt": 20,
}

label_SIRI_WHU = {
    "agriculture": 0,
    "commercial": 1,
    "harbor": 2,
    "idle_land": 3,
    "industrial": 4,
    "meadow": 5,
    "overpass": 6,
    "park": 7,
    "pond": 8,
    "residential": 9,
    "river": 10,
    "water": 11,
}

label_mapping = {
    "UCMerced_LandUse": label_UCMerced_LandUse,
    "SIRI_WHU":label_SIRI_WHU,
}


class customFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(customFolder, self).__init__(root, transform)
        self.path_list = [path for path, target in self.samples]

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path

    def __len__(self):
        return super().__len__()


data_transform_224 = transforms.Compose(
    [transforms.Resize([224, 224]), transforms.ToTensor()]
)

data_transform_299 = transforms.Compose(
    [transforms.Resize([299, 299]), transforms.ToTensor()]
)


class trainDataset(data.Dataset):
    def __init__(self, model: str, args) -> None:
        """_summary_

        Args:
            model (_type_): 'train' or 'test'
            args (_type_): argparse
        """
        #self.model_name = model_name
        self.data_dir = os.path.join(args.data_dir, args.data_type, model)
        self.data_path = []
        self.data_label = []
        self.label_box = label_mapping[args.data_type]
        
        # 检查数据集路径是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据集路径不存在: {self.data_dir}\n请检查:\n1. data_dir是否正确: {args.data_dir}\n2. data_type是否正确: {args.data_type}\n3. 是否存在 {model} 文件夹")
        
        # 检查每个类别文件夹
        found_classes = []
        missing_classes = []
        for subdir in self.label_box.keys():
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                found_classes.append(subdir)
                # 获取子文件夹中的文件名
                file_count = 0
                for file_path in glob.glob(os.path.join(subdir_path, "*")):
                    if os.path.isfile(file_path):  # 只添加文件，不添加文件夹
                        self.data_path.append(file_path)
                        self.data_label.append(subdir)
                        file_count += 1
                if file_count == 0:
                    print(f"警告: 类别文件夹 {subdir} 存在但没有找到图片文件")
            else:
                missing_classes.append(subdir)
        
        if len(self.data_path) == 0:
            raise ValueError(
                f"数据集为空！\n"
                f"数据集路径: {self.data_dir}\n"
                f"找到的类别文件夹: {found_classes}\n"
                f"缺失的类别文件夹: {missing_classes}\n"
                f"请检查数据集目录结构是否正确"
            )
        
        print(f"成功加载数据集: {len(self.data_path)} 张图片, {len(found_classes)} 个类别")

    def __getitem__(self, index):
        image_path = self.data_path[index]
        image = PIL.Image.open(image_path).convert("RGB")
        image = data_transform_224(image)
        label = self.label_box[self.data_label[index]]
        return image, label

    def __len__(self):
        return len(self.data_path)


class attackDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(args.data_dir, args.data_type, "test")
        csv_path = "/home/byf/Adversarial-Attack/logs/{}_list.csv".format(args.data_type)
        if not os.path.exists(csv_path):
            print("no the list of data, build new list")
            get_images_lists(args.data_dir, args.data_type, args.device)

        label_box = os.listdir(self.data_dir)
        self.label_dict = label_mapping[args.data_type]
        self.name_dict = {
            self.label_dict[string]: string for string in self.label_dict.keys()
        }

        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path, index_col=0)
            self.path_box = self.df.columns.to_list()
        else:
            raise Exception(
                "no file name {}, please run \
                            the code of get_images_lists()".format(
                    csv_path
                )
            )

    def __getitem__(self, index):
        image_path = (
            os.path.join(self.args.data_dir, self.args.data_type, "test")
            + self.path_box[index]
        )
        image_path = image_path.replace("/", os.sep)
        image_path = image_path.replace("\\", os.sep)
        image = PIL.Image.open(image_path).convert("RGB")
        
        image = data_transform_224(image)

        (dir_name, file_name) = os.path.split(image_path)
        (_, label_name) = os.path.split(dir_name)

        te = torch.tensor(self.label_dict[label_name])

        return image, te, file_name

    def __len__(self):
        return len(self.df.columns)


class evalDataset(attackDataset):
    def __init__(self, adv_dir, args):
        super(evalDataset, self).__init__(args)
        self.adv_dir = adv_dir

    def __getitem__(self, index):
        image_path = (
            os.path.join(self.args.data_dir, self.args.data_type, "test")
            + self.path_box[index]
        )
        image_path = image_path.replace("/", os.sep)
        image_path = image_path.replace("\\", os.sep)
        image_clean = PIL.Image.open(image_path).convert("RGB")
        image_clean = data_transform_224(image_clean)

        (dir_name, file_name) = os.path.split(image_path)
        (_, label_name) = os.path.split(dir_name)

        adv_path = os.path.join(self.adv_dir, label_name, file_name)

        image_adv = PIL.Image.open(adv_path).convert("RGB")
        
        image_adv = data_transform_224(image_adv)

        return image_clean, image_adv


def get_images_lists(data_dir, data_type, device="cuda:0"):
    """Obtain a list of images that are correctly recognized across all images."""
    data_dir = os.path.join(data_dir, data_type, "test")
    dataset = customFolder(data_dir, data_transform_224)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=5)
    paths = dataset.path_list
    paths = [path.replace(data_dir, "") for path in paths]
    paths = [path.replace("\\", "/") for path in paths]
    df = pd.DataFrame(columns=paths)

    for key, fc in model_box.items():
        
        model_name = key + "_" + data_type + ".pt"
        model_path = os.path.join("/home/byf/Adversarial-Attack/checkpoints", model_name)
        model_dict = torch.load(model_path, map_location="cpu")
        if data_type == "FGSCR_42":
            num_classes = 42
        elif data_type == "MTARSI":
            num_classes = 20
        elif data_type == "UCMerced_LandUse":
            num_classes = 21
        elif data_type == "SIRI_WHU":
            num_classes = 12
        net = fc(num_classes)
        net.load_state_dict(model_dict)
        net = net.to(device)
        net.eval()
        for data, true_label, paths in loader:
            paths = [path.replace(data_dir, "") for path in paths]
            paths = [path.replace("\\", "/") for path in paths]
            df.loc[key, paths] = true_label.tolist()
        with torch.no_grad():
            for data, true_label, paths in tqdm(loader):
                paths = [path.replace(data_dir, "") for path in paths]
                paths = [path.replace("\\", "/") for path in paths]
                data, true_label = data.to(device), true_label.to(device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                df.loc[key, paths] = pred.tolist()

    for image_path in df.columns:
        if not df[image_path].nunique() == 1:
            df = df.drop(columns=image_path)
    df.to_csv("/home/byf/Adversarial-Attack/logs/{}_list.csv".format(data_type))
