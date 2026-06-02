import math
import torch
import random
from PIL import ImageOps,Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torch.nn import Dropout
import copy
import pdb
import numpy as np
from typing import Union, List, Optional, Tuple, Any
import io
import pytorch_wavelets


def get_p(aug_param_group, aug_param_method, aug_param_strength):
    raw_group   = torch.nan_to_num(aug_param_group)
    raw_group = raw_group - raw_group.max()
    p_g = torch.softmax(raw_group, dim=0)
    
    p_m = []
    for i in range(len(group_trans_lists)):
        aug_param_method_i = aug_param_method[i]
        raw_method = torch.nan_to_num(aug_param_method_i)
        raw_method = raw_method - raw_method.max()
        p_m.append(torch.softmax(raw_method, dim=0))

    p_s = []
    for i in range(len(group_trans_lists)):
        p_s_n = []
        for j in range(len(group_trans_lists[str(i)])):
            # 安全处理：去掉 NaN/Inf，并做平移后再 softmax，避免出现非法概率
            raw_strength = torch.nan_to_num(aug_param_strength[i][j])
            raw_strength = raw_strength - raw_strength.max()
            p_s_n.append(torch.softmax(raw_strength, dim=0))
        p_s.append(p_s_n)
    return p_g, p_m, p_s

def get_translists_ids(l1, l2, l3, p_g, p_m, p_s, ops_num):
    translist_ids = []
    for i in range(l1):
        group_id = torch.multinomial(p_g, 1, replacement=True)
        for j in range(l2):
            method_id = torch.multinomial(p_m[group_id], 1, replacement=True)
            for k in range(l3):
                strength_id = torch.multinomial(p_s[group_id][method_id], 1, replacement=True)
                translist_ids.append([[group_id.item(), method_id.item(), strength_id.item()]])
    translist_ids = np.array(translist_ids)
    np.random.shuffle(translist_ids)
    trans_num = int(len(translist_ids)/ops_num)
    translist_ids = translist_ids.reshape(trans_num, ops_num, 3)
    return translist_ids

def get_prob(p_g, p_m, p_s,group_id, method_id, strength_id,ops_num):
    tp=1
    for i in range(ops_num):
        tp = tp * p_g[group_id[i]] * p_m[group_id[i]][method_id[i]] * p_s[group_id[i]][method_id[i]][strength_id[i]]
    return tp   


class RWAug_Search_H: 
    def __init__(self, n, group_ids, method_ids, strength_ids):
        self.n = n
        #idxs is the operation id
        self.group_ids = group_ids
        self.method_ids = method_ids
        self.strength_ids = strength_ids

    def __call__(self, img):
      assert len(self.group_ids) == self.n
      assert len(self.method_ids) == self.n
      for i in range(self.n):
        group_id =  self.group_ids[i].item()
        method = self.method_ids[i]
        strength = self.strength_ids[i]+1
        func = group_trans_lists[str(group_id)][int(method)](strength=strength,num_scale=4)
        img = func(img)
      return img

class identity:
    def __init__(self,strength=1,num_scale=4) -> None:
        self.strength = strength
        self.num_scale = num_scale

    def __call__(self, x):
        return x.repeat(self.num_scale,1,1,1)

class affine:#平移变换
    def __init__(self, offset=0.1, num_scale=5, strength= 1) -> None:
        self.num_scale = num_scale
        self.offset = offset
        if strength == 0: self.offset = 0.4
        if strength == 1: self.offset = 0.5
        if strength == 2: self.offset = 0.6
        if strength == 3: self.offset = 0.7
        if strength == 4: self.offset = 0.8
        if strength == 5: self.offset = 0.9
        
    def __call__(self, x):
        return torch.cat([transforms.functional.affine(img=x, angle=0, translate=[self.offset*(i+1)/self.num_scale, self.offset*(i+1)/self.num_scale], scale=1, shear=0) for i in range(self.num_scale)])

class rotate():#旋转变换
    def __init__(self, angle=30, num_scale=5, strength=1) -> None:
        self.num_scale = num_scale
        self.angle = angle
        if strength == 0: self.angle = 0
        if strength == 1: self.angle = 30
        if strength == 2: self.angle = 60
        if strength == 3: self.angle = 90
        if strength == 4: self.angle = 120
        if strength == 5: self.angle = 150
    
    def __call__(self, x):
        return torch.cat([transforms.functional.rotate(img=x, angle=(self.angle / (2**i))) for i in range(self.num_scale)])

class crop():#裁剪
    def __init__(self, ratio=0.1,  num_scale=4, strength=1) -> None:
        self.num_scale = num_scale
        self.ratio = ratio
        if strength == 0: self.ratio = 0
        if strength == 1: self.ratio = 0.1
        if strength == 2: self.ratio = 0.3
        if strength == 3: self.ratio = 0.5
        if strength == 4: self.ratio = 0.7
        if strength == 5: self.ratio = 0.9

    def crop(self, x, ratio):
        width = int(x.shape[2]*ratio)
        height = int(x.shape[3]*ratio)
        
        left = 0+(x.shape[2]-width)//2
        top = 0+(x.shape[3]-height)//2
        return transforms.functional.resized_crop(x, top, left, height, width, (224, 224))
        
    def __call__(self, x) -> Any:
        #transforms.functional.resized_crop(x, 0, 0, int(0.9*224), int(0.9*224), (224, 224))
        return torch.cat([self.crop(x, self.ratio+(1-self.ratio)*(i+1)/self.num_scale) for i in range(self.num_scale)])

class dim():
    def __init__(self, resize_rate=1.1, diversity_prob=0.5, num_scale=4, strength=1) -> None:
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.num_scale = num_scale
        if strength == 0: self.resize_rate = 1
        if strength == 1: self.resize_rate = 1.1
        if strength == 2: self.resize_rate = 1.2
        if strength == 3: self.resize_rate = 1.3
        if strength == 4: self.resize_rate = 1.4
        if strength == 5: self.resize_rate = 1.5
        
    def apply_once(self, x):
        # do not transform the input image
        #if torch.rand(1) > self.diversity_prob:
        #    return x
        if self.resize_rate==1:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    
    def __call__(self, x):
        outs = []
        for i in range(self.num_scale):
            outs.append(self.apply_once(x))
        return torch.cat(outs, dim=0)

class blur():  # 高斯模糊
    def __init__(self, ksize: Union[int, List[int]] = 5,
                 sigma_min: float = 0.1,
                 sigma_max: float = 3.0,
                 num_scale: int = 4,
                 strength: float = 1) -> None:
        
        if isinstance(ksize, int):
            if ksize % 2 == 0:
                ksize += 1
            self.ksize = [ksize, ksize]
        else:
            assert len(ksize) == 2 and ksize[0] > 0 and ksize[1] > 0
            self.ksize = [int(ksize[0] // 2 * 2 + 1), int(ksize[1] // 2 * 2 + 1)]
        
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.num_scale = int(num_scale)
        self.strength = max(0, min(5, float(strength)))

    def _apply_once(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma == 0:
            return x.clone()
        y = transforms.functional.gaussian_blur(x, kernel_size=self.ksize, sigma=[sigma, sigma])
        return torch.clamp(y, 0.0, 1.0)

    def __call__(self, x: torch.Tensor) -> Any:
        if self.strength == 0:
            return x.clone()
        
        # 根据strength调整sigma范围
        scale_factor = self.strength / 5.0  # 映射到0-1范围
        adjusted_sigma_min = self.sigma_min * scale_factor
        adjusted_sigma_max = self.sigma_max * scale_factor
        
        # 生成多尺度模糊
        outs = []
        for i in range(self.num_scale):
            t = (i + 1) / self.num_scale
            sigma = adjusted_sigma_min + (adjusted_sigma_max - adjusted_sigma_min) * t
            outs.append(self._apply_once(x, sigma))
        
        return torch.cat(outs, dim=0)

class ide():#
    def __init__(self, dropout_prob=[0,0.1,0.2,0.3],strength=1,num_scale=4) -> None:
        self.dropout_prob = dropout_prob
        if strength == 0: self.dropout_prob = [0,0,0,0]
        if strength == 1: self.dropout_prob = [0,0.1,0.2,0.3]
        if strength == 2: self.dropout_prob = [0.1,0.2,0.3,0.4]
        if strength == 3: self.dropout_prob = [0.2,0.3,0.4,0.5]
        if strength == 4: self.dropout_prob = [0.3,0.4,0.5,0.6]
        if strength == 5: self.dropout_prob = [0.4,0.5,0.6,0.7]
        
    def __call__(self, x):
        return torch.cat([Dropout(p=prob)(x)*(1-prob) for prob in self.dropout_prob])

class sim():
    def __init__(self, num_scale=3, strength=1) -> None:
        self.num_scale = strength+3
    
    def __call__(self, x):
        return torch.cat([x / (2**i) for i in range(self.num_scale)])

class admix():
    def __init__(self, num_admix=3, num_scale=3, strength=1) -> None:
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.strength = max(0, min(5, float(strength)))
        self.admix_strength = self.strength / 10.0  # 将strength映射到0-1范围

    def __call__(self, x) -> Any:
        admix_images = torch.concat([(x + self.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_admix)], dim=0)
        return torch.concat([admix_images / (2 ** i) for i in range(self.num_scale)])

class blockshuffle():#块级重组
    def __init__(self, num_block=3, num_scale=4, strength=1) -> None:
        self.num_scale = num_scale
        self.num_block = strength + 2
        
    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        # perm = torch.randperm(self.num_block)
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def __call__(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])
    
class blockrotate():
    def __init__(self, 
                 grid_size: int = 3,  # 网格大小 (n x n)
                 rotate_blocks: int = 4,  # 要旋转的块数量
                 angle: int = 30,  # 最大旋转角度
                 num_scale: int = 5,  # 尺度数量
                 strength: int = 1  # 旋转强度
                 ) -> None:
        
        self.grid_size = grid_size
        self.rotate_blocks = min(rotate_blocks, grid_size * grid_size)
        self.num_scale = num_scale
        
        # 根据strength设置角度
        if strength == 0: self.angle = 0
        elif strength == 1: 
            self.angle = 30 
            self.grid_size=2
            self.rotate_blocks=1
        elif strength == 2: 
            self.angle = 60
            self.grid_size=2
            self.rotate_blocks=3
        elif strength == 3: 
            self.angle = 90
            self.grid_size=3
            self.rotate_blocks=1
        elif strength == 4: 
            self.angle = 120
            self.grid_size=2
            self.rotate_blocks=3
        elif strength == 5: 
            self.angle = 150
            self.grid_size=2
            self.rotate_blocks=5
    
    def _split_into_blocks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """将图片分割成n x n个块"""
        _, _, H, W = x.shape
        block_h = H // self.grid_size
        block_w = W // self.grid_size
        
        blocks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                h_start = i * block_h
                h_end = (i + 1) * block_h
                w_start = j * block_w
                w_end = (j + 1) * block_w
                
                block = x[:, :, h_start:h_end, w_start:w_end]
                blocks.append(block)
        
        return blocks
    
    def _reconstruct_from_blocks(self, blocks: List[torch.Tensor]) -> torch.Tensor:
        """从块重建图片"""
        _, C, block_h, block_w = blocks[0].shape
        H = block_h * self.grid_size
        W = block_w * self.grid_size
        
        reconstructed = torch.zeros((1, C, H, W), dtype=blocks[0].dtype, device=blocks[0].device)
        
        for idx, block in enumerate(blocks):
            i = idx // self.grid_size
            j = idx % self.grid_size
            h_start = i * block_h
            h_end = (i + 1) * block_h
            w_start = j * block_w
            w_end = (j + 1) * block_w
            
            reconstructed[:, :, h_start:h_end, w_start:w_end] = block
        
        return reconstructed
    
    def _rotate_block(self, block: torch.Tensor, angle: float) -> torch.Tensor:
        """旋转单个块"""
        if angle == 0:
            return block.clone()
        
        # 旋转块
        rotated_block = TF.rotate(block.squeeze(0), angle)
        return rotated_block.unsqueeze(0)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """应用块旋转变换"""
        if self.angle == 0:
            return x.repeat(self.num_scale, 1, 1, 1)
        
        results = []
        
        for scale_idx in range(self.num_scale):
            # 计算当前尺度的旋转角度
            current_angle = self.angle
            
            # 分割图片为块
            blocks = self._split_into_blocks(x)
            
            # 随机选择要旋转的块
            block_indices = list(range(len(blocks)))
            random.shuffle(block_indices)
            rotate_indices = block_indices[:self.rotate_blocks]
            
            # 对选中的块进行旋转
            rotated_blocks = []
            for idx, block in enumerate(blocks):
                if idx in rotate_indices:
                    rotated_block = self._rotate_block(block, current_angle)
                else:
                    rotated_block = block.clone()
                rotated_blocks.append(rotated_block)
            
            # 重建图片
            reconstructed = self._reconstruct_from_blocks(rotated_blocks)
            results.append(reconstructed)
        
        return torch.cat(results, dim=0)

class blockmix():
    def __init__(self, strength=1, num_scale=4) -> None:
        self.num_scale = num_scale
        self.split_num = strength + 2
        self.mix_strength = strength  * 0.2 - 0.1

    def get_length(self, length):
        rand = np.random.uniform(size=self.split_num)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        # perm = torch.randperm(self.num_block)
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])
    
    def _apply_once(self, x):

        mixing_x = self.shuffle(x)
        blockmixed_x = x + self.mix_strength * mixing_x

        return blockmixed_x

    def __call__(self, x):
        
        return torch.cat([self._apply_once(x) for _ in range(self.num_scale)])

class masked():
    def __init__(self, num_block=3, num_scale=4, strength=1) -> None:
        self.num_scale = num_scale
        self.num_block = strength + 2
        if strength ==1 : self.masked_block_num = 1
        if strength ==2 : self.masked_block_num = 3
        if strength ==3 : self.masked_block_num = 5
        if strength ==4 : self.masked_block_num = 7
        if strength ==5 : self.masked_block_num = 9

    def blockmask(self, x, choice=-1):
        _, _, w, h = x.shape
        
        if w == h:
            step = w / self.num_block
            points = [round(step * i) for i in range(self.num_block + 1)]
        
        x_copy = x.clone()
        for i in range(self.masked_block_num):
            x_block, y_block = random.randint(0, self.num_block-1), random.randint(0, self.num_block-1)
            x_copy[:, :, points[x_block]:points[x_block+1], points[y_block]:points[y_block+1]] = 0
        
        return x_copy
    
    def __call__(self, x):
        return torch.cat([self.blockmask(x) for _ in range(self.num_scale)])

class wrap:
    def __init__(self, strength=1, num_scale=4):
        self.noise_scale = 2
        self.mesh_width = 3
        self.mesh_height = 3
        self.rho = 0.01
        self.num_scale = num_scale
        if strength==0: self.noise_scale = 0
        if strength==1: self.noise_scale = 0.5
        if strength==2: self.noise_scale = 1
        if strength==3: self.noise_scale = 2
        if strength==4: self.noise_scale = 3
        if strength==5: self.noise_scale = 4

    def K_matrix(self, X, Y):
        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)
        return K

    def P_matrix(self, X):
        n, k = X.shape[:2]
        device = X.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = X
        return P

    def tps_coeffs(self, X, Y):
        n, k = X.shape[:2]
        device = X.device

        # 构建线性方程组 L * Q = Z
        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)
    
        # 计算核矩阵
        K = self.K_matrix(X, X)
    
        # 填充矩阵
        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # 求解线性方程组
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]

    def tps(self, source_points, target_points, image_size=(256, 256), grid=None, device=None):
        h, w = image_size
    
        # 如果没有提供网格，则创建
        if grid is None:
            if device is None:
                device = source_points.device
            grid = torch.ones(1, h, w, 2, device=device)
            grid[:, :, :, 0] = torch.linspace(-1, 1, w, device=device)
            grid[:, :, :, 1] = torch.linspace(-1, 1, h, device=device)[..., None]
            grid = grid.view(-1, h * w, 2)
    
    # 计算TPS系数
        W, A = self.tps_coeffs(source_points, target_points)
    
        # 应用变换到整个网格
        U = self.K_matrix(grid, source_points)
        P = self.P_matrix(grid)
        warped_grid = P @ A + U @ W
    
        return warped_grid.view(-1, h, w, 2)

    def grid_points_2d(self, width, height, device):
        xx, yy = torch.meshgrid(
            [torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device)])
        return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)

    def noisy_grid(self, width, height, noise_map, device):
        """
        Make uniform grid points, and add noise except for edge points.
        """
        grid = self.grid_points_2d(width, height, device)
        mod = torch.zeros([height, width, 2], device=device)
        mod[1:height - 1, 1:width - 1, :] = noise_map
        return grid + mod.reshape(-1, 2)

    def _apply_once(self, x):
        noise_map = (torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2]) - 0.5) * self.noise_scale
        n, c, w, h = x.size()
        device = x.device
        X = self.grid_points_2d(self.mesh_width, self.mesh_height, device)
        Y = self.noisy_grid(self.mesh_width, self.mesh_height, noise_map, device)
        warped_grid_b = self.tps(source_points=X[None, ...], target_points=Y[None, ...], image_size=(h, w), device=device)
        warped_grid_b = warped_grid_b.repeat(x.shape[0], 1, 1, 1)
        vwt_x = torch.grid_sampler_2d(x, warped_grid_b, 0, 0, False)
        return vwt_x  

    def __call__(self, x):
        outs=[]
        for i in range(self.num_scale):
            outs.append(self._apply_once(x))
        return torch.cat(outs, dim=0)

class ssm():
    def __init__(self, strength=1,num_scale=4):
        self.strength = max(0, min(9, int(round(strength))))  # 取整到0-5
        
        # 根据不同强度档位设置不同的扰动参数
        self.strength_configs = {
            0: {"epsilon": 16/255, "rho": 0.0},  
            1: {"epsilon": 16/255, "rho": 0.1}, 
            2: {"epsilon": 16/255, "rho": 0.3},  
            3: {"epsilon": 16/255, "rho": 0.5},  
            4: {"epsilon": 16/255, "rho": 0.7},  
            5: {"epsilon": 16/255, "rho": 0.9}   

        }
        
        config = self.strength_configs[self.strength]
        self.epsilon = config["epsilon"]
        self.rho = config["rho"]
        self.num_scale = num_scale

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v)
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)
        return V

    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]
        X_v = X.contiguous().view(-1, x_shape[-1]) / 2
        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2
        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r
        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape).real

    def dct_2d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
    
    def __call__(self, x):
        if self.strength == 0:
            # strength=0时返回原图
            return x.repeat(self.num_scale, 1, 1, 1)
        
        x_idct = []
        device = x.device
        
        for _ in range(self.num_scale):
            gauss = torch.randn(x.size()[0], 3, 224, 224) * self.epsilon
            gauss = gauss.to(device)
            x_dct = self.dct_2d(x + gauss).to(device)
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).to(device)
            x_idct.append(self.idct_2d(x_dct * mask))

        return torch.cat(x_idct)

class dwt_mask():
    def __init__(self, rho=0.1, num_scale=4, wave="db3",strength=1) -> None:
        self.num_scale = num_scale
        self.wave = wave
        if strength==0: self.rho = 0
        if strength==1: self.rho = 0.1
        if strength==2: self.rho = 0.2
        if strength==3: self.rho = 0.3
        if strength==4: self.rho = 0.4
        if strength==5: self.rho = 0.5

    def _apply_once(self, x, rho):
        device = x.device
        xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wave, mode="zero").to(device)
        Yl, Yh = xfm(x)

        # 在低频部分加 mask
        maskl = torch.rand_like(Yl) * (2 * rho) + (1 - rho)
        Yl = Yl * maskl.to(device)
        # 在高频部分加 mask
        maskh = []
        for i  in range(len(Yh)):
            maskh.append(torch.rand_like(Yh[i]) * (2 * rho) + (1 - rho))
            Yh[i] = Yh[i] * maskh[i].to(device)

        ifm = pytorch_wavelets.DWTInverse(wave=self.wave, mode="zero").to(device)
        Y = ifm((Yl, Yh))
        return Y

    def __call__(self, x):
        outs = []
        for i in range(self.num_scale):
            outs.append(self._apply_once(x, self.rho))
        return torch.cat(outs, dim=0)

class low_freq_mask():
    def __init__(self, rho=0.1, num_scale=4, wave="db3",strength=1) -> None:
        self.num_scale = num_scale
        self.wave = wave
        if strength==0: self.rho = 0
        if strength==1: self.rho = 0.1
        if strength==2: self.rho = 0.2
        if strength==3: self.rho = 0.3
        if strength==4: self.rho = 0.4
        if strength==5: self.rho = 0.5

    def _apply_once(self, x, rho):
        device = x.device
        xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wave, mode="zero").to(device)
        Yl, Yh = xfm(x)

        # 在低频部分加 mask
        maskl = torch.rand_like(Yl) * (2 * rho) + (1 - rho)
        Yl = Yl * maskl.to(device)

        ifm = pytorch_wavelets.DWTInverse(wave=self.wave, mode="zero").to(device)
        Y = ifm((Yl, Yh))
        return Y

    def __call__(self, x):
        outs = []
        for i in range(self.num_scale):
            outs.append(self._apply_once(x, self.rho))
        return torch.cat(outs, dim=0)

class high_freq_mask():
    def __init__(self, rho=0.1, num_scale=4, wave="db3",strength=1) -> None:
        self.num_scale = num_scale
        self.wave = wave
        if strength==0: self.rho = 0
        if strength==1: self.rho = 0.1
        if strength==2: self.rho = 0.2
        if strength==3: self.rho = 0.3
        if strength==4: self.rho = 0.4
        if strength==5: self.rho = 0.5

    def _apply_once(self, x, rho):
        device = x.device
        xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wave, mode="zero").to(device)
        Yl, Yh = xfm(x)

        device = x.device
        xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wave, mode="zero").to(device)
        Yl, Yh = xfm(x)

        # 在高频部分加 mask
        maskh = []
        for i  in range(len(Yh)):
            maskh.append(torch.rand_like(Yh[i]) * (2 * rho) + (1 - rho))
            Yh[i] = Yh[i] * maskh[i].to(device)

        ifm = pytorch_wavelets.DWTInverse(wave=self.wave, mode="zero").to(device)
        Y = ifm((Yl, Yh))
        return Y

    def __call__(self, x):
        outs = []
        for i in range(self.num_scale):
            outs.append(self._apply_once(x, self.rho))
        return torch.cat(outs, dim=0)

class freq_band_dropout():
    def __init__(self, strength: float = 1.0, num_scale: int = 4):
        self.strength = max(0, min(5, int(round(strength))))  # 取整到0-5
        self.num_scale = num_scale
        
        self.strength_configs = {
            0: {"drop_range": (0.0, 0.0), "description": "无频带丢弃"},
            1: {"drop_range": (0.1, 0.3), "description": "轻微频带丢弃"},
            2: {"drop_range": (0.2, 0.4), "description": "中等频带丢弃"},
            3: {"drop_range": (0.3, 0.5), "description": "较强频带丢弃"},
            4: {"drop_range": (0.4, 0.6), "description": "强烈频带丢弃"},
            5: {"drop_range": (0.5, 0.7), "description": "最强频带丢弃"}
        }

    def _apply_once(self, x, drop_min, drop_max):
        """
        对单一强度档执行随机频带丢弃
        """
        B, C, H, W = x.shape
        device = x.device

        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij"
        )
        dist = torch.sqrt(xx**2 + yy**2)  # 归一化频率半径 [0, 1]

        drop_center = torch.rand(1, device=device).item() * (1.0 - drop_max)
        drop_start = drop_center
        drop_end = drop_center + (drop_max - drop_min) * torch.rand(1, device=device).item() + drop_min

        mask = (dist < drop_start) | (dist > drop_end)
        mask = mask.to(Xf.dtype)

        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        Xf = Xf * mask

        Xf = torch.fft.ifftshift(Xf)
        x_out = torch.fft.ifft2(Xf, norm="ortho").real
        return x_out.clamp(0.0, 1.0)

    def __call__(self, x):
        if self.strength == 0:
            return x.repeat(self.num_scale, 1, 1, 1)
        
        config = self.strength_configs[self.strength]
        drop_min, drop_max = config["drop_range"]
        
        outs = []
        for i in range(self.num_scale):
            current_drop = drop_min + (drop_max - drop_min) * (i / max(1, self.num_scale - 1))
            outs.append(self._apply_once(x, current_drop, current_drop))
        
        return torch.cat(outs, dim=0)

class freq_enhancer_trans():
    def __init__(self, boost=0.1, num_scale=4, wave="db3", strength=1) -> None:
        self.num_scale = num_scale
        self.wave = wave
        
        # 强度配置
        strength_configs = {
            0: {"boost": 0.0, "description": "无频域增强"},
            1: {"boost": 0.1, "description": "轻微频域增强"},
            2: {"boost": 0.2, "description": "中等频域增强"},
            3: {"boost": 0.3, "description": "强频域增强"},
            4: {"boost": 0.4, "description": "很强频域增强"},
            5: {"boost": 0.5, "description": "最强频域增强"}
        }
        
        config = strength_configs.get(strength, strength_configs[1])
        self.boost = config["boost"]

    def _apply_once(self, x, boost):
        device = x.device
        xfm = pytorch_wavelets.DWTForward(J=1, wave=self.wave, mode="zero").to(device)
        Yl, Yh = xfm(x)

        # 增强低频部分 - 使用乘法增强
        if boost > 0:
            Yl = Yl * (1 + boost)
        else:
            # 如果是负增强，可以用于减弱低频
            Yl = Yl * (1 + boost)

        # 增强高频部分 - 对每个方向的高频分量进行增强
        enhanced_Yh = []
        for band in Yh:
            # 对每个高频子带进行增强
            enhanced_band = band * (1 + boost)
            enhanced_Yh.append(enhanced_band)

        ifm = pytorch_wavelets.DWTInverse(wave=self.wave, mode="zero").to(device)
        Y = ifm((Yl, enhanced_Yh))
        return Y

    def __call__(self, x):
        if self.boost == 0:
            return x.repeat(self.num_scale, 1, 1, 1)
            
        outs = []
        for i in range(self.num_scale):
            outs.append(self._apply_once(x, self.boost))
        return torch.cat(outs, dim=0)

class freq_compress_trans():
    def __init__(self, compress_ratio=0.8, num_scale=4, strength=1) -> None:
        self.num_scale = num_scale
        
        # 根据strength设定压缩比例
        if strength == 0: self.compress_ratio = 1.0
        if strength == 1: self.compress_ratio = 2.0
        if strength == 2: self.compress_ratio = 1.8
        if strength == 3: self.compress_ratio = 1.6
        if strength == 4: self.compress_ratio = 1.4
        if strength == 5: self.compress_ratio = 1.2

    def _apply_once(self, x, compress_ratio):
        # FFT变换
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # 获取目标尺寸
        B, C, H, W = x_fft_shifted.shape
        new_H, new_W = int(H * compress_ratio), int(W * compress_ratio)
        new_H, new_W = max(1, new_H), max(1, new_W)  # 确保至少1x1
        
        # 分别对实部和虚部进行插值
        real_part = x_fft_shifted.real
        imag_part = x_fft_shifted.imag
        
        # 下采样实部
        compressed_real = F.interpolate(
            real_part, 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 下采样虚部
        compressed_imag = F.interpolate(
            imag_part, 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 重新组合为复数
        compressed_fft = torch.complex(compressed_real, compressed_imag)
        
        # 上采样回原尺寸
        restored_real = F.interpolate(
            compressed_real,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        restored_imag = F.interpolate(
            compressed_imag,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        restored_fft = torch.complex(restored_real, restored_imag)
        restored_fft_shifted = torch.fft.ifftshift(restored_fft, dim=(-2, -1))
        
        # 逆FFT
        x_restored = torch.fft.ifft2(restored_fft_shifted, norm='ortho').real
        
        return x_restored

    def __call__(self, x):
        outs = []
        for i in range(self.num_scale):
            outs.append(self._apply_once(x, self.compress_ratio))
        return torch.cat(outs, dim=0)

spatial_global_list = [affine,rotate,dim,sim]
spatial_local_list = [blockshuffle,masked,blockmix,wrap]
frequency_global_list = [ssm,blur,ssm,dwt_mask]
frequency_local_list = [low_freq_mask,freq_band_dropout,high_freq_mask,freq_band_dropout] 


group_trans_lists = {
    "0": spatial_global_list,
    "1": spatial_local_list,
    "2": frequency_global_list,
    "3": frequency_local_list
}
