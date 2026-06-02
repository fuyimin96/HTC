import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from rsiattack import ATTACK
from .utils.HTC_transform import RWAug_Search_H, group_trans_lists, RWAug_Search_3, get_translists_ids, get_p, get_prob
import numpy as np
import math

import sys

class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self, eps=1e-8, coeff=5.0):
        super(Mid_layer_target_Loss, self).__init__()
        self.eps = eps
        self.coeff = coeff

    def forward(self, old_attack_mid, new_mid, original_mid):
        x = (old_attack_mid - original_mid).reshape(old_attack_mid.size(0), -1)  
        y = (new_mid - original_mid).reshape(new_mid.size(0), -1)                

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)                       
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)                    

        cosine_sim = F.cosine_similarity(x, y, dim=1, eps=self.eps)            

        angle_loss = cosine_sim.mean()              
        return angle_loss

class HTC(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        parser = self.create_options(parser)
        self.args = self.add_options(parser)

        super(HTC, self).__init__(self.args)

    def add_options(self, parser):
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--eps", type=float, default=16)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--mu", type=float, default=1.0)
        parser.add_argument("--L_1", type=int, default=2)
        parser.add_argument("--L_2", type=int, default=2)
        parser.add_argument("--L_3", type=int, default=2)
        parser.add_argument("--xi", type=float, default=0.2)
        parser.add_argument("--lambda_", type=str, default=0.5)
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args
    
    def get_loss(self, logits, label, num_copy):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        return self.loss(logits, label.repeat(num_copy))

    def get_grad(self, loss, delta, **kwargs):
        if not loss.requires_grad:
            raise RuntimeError("loss.requires_grad=False:loss被detach或在no_grad里构造了")
        if not delta.requires_grad:
            delta = delta.detach().requires_grad_(True)
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, allow_unused=False)[0]
        return grad

    def transform(self, x, **kwargs):
        return kwargs['search'](x)

    def get_hook(self, model):

        model_name = model.__class__.__name__

        if model_name == 'ResNet':
            return model.layer4[2].bn2
        if model_name == 'VGG':
            return model.features[-1]
        if model_name == 'DenseNet':
            return model.features[-1].norm5
        if model_name == 'Inception_ResNetv2':
            return model.features[-1]
        if model_name == 'GoogLeNet':
            return model.inception5b   
        if model_name == 'Inception3':
            return model.Mixed_7c.branch3x3dbl_3b.bn
        if model_name == 'VisionTransformer':
            return model.encoder.layers[-1]

        else :
            print("model_name is not supported")
            return None

    def get_logits_features(self, net, images, require_grad: bool = True):  # 默认设为True
        device = self.args.device
        mid_output = None

        def get_mid_output(module, input, output):
            nonlocal mid_output
            mid_output = output  

        feature_layer = self.get_hook(net)
        hook = feature_layer.register_forward_hook(get_mid_output)

        if net.__class__.__name__ == "Inception3":
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        if not images.requires_grad:
            images = images.requires_grad_(True)

        try:
            logits = net(images)
            if mid_output is None:
                raise RuntimeError("Hook did not capture any output!")
            return logits.to(device), mid_output.to(device)
        finally:
            hook.remove()
    
    def l2t_forward_H(self, images, label, **kwargs): 
        args = self.args
        device = args.device
        net = self.net.to(device)
        ops_num = args.L_1
        epsilon = args.eps / 255
        alpha = args.alpha / 255

        aug_param_group = torch.nn.Parameter(torch.ones(len(group_trans_lists)),requires_grad=True)
        aug_param_methods = torch.nn.Parameter(torch.ones(len(group_trans_lists), 4, requires_grad=True))
        aug_param_strength = torch.nn.Parameter(torch.ones(len(group_trans_lists), 4, 5, requires_grad=True))

        PPO_optimizer = torch.optim.Adam([aug_param_group, aug_param_methods, aug_param_strength], lr=0.03)

        label = label.clone().detach().to(device)
        delta = torch.zeros_like(images).requires_grad_(True).to(device)

        #计算上一轮中间层的输出
        used_delta = delta
        last_g = 0
        mid_original = 0
        mid_attack_orignal = 0

        _, mid_original = self.get_logits_features(net, images,require_grad=True)

        for e in range(args.epochs):
            delta = delta.detach().clamp(-epsilon, epsilon).requires_grad_(True)
            # transform data
            aug_probs = []
            losses = []
            mid_losses = []

            used_E_loss = 0
            used_E_mid_loss = 0
            lamda = args.lambda_

            _, mid_attack_original = self.get_logits_features(net, images+used_delta,require_grad=True)
            

            p_g,p_m,p_s = get_p(aug_param_group, aug_param_methods, aug_param_strength)
            trans_ids_lists = get_translists_ids(args.L_1,args.L_2,args.L_3,p_g,p_m,p_s,ops_num)
            num_scale = len(trans_ids_lists)
            for i in range(num_scale): 
                rw_search = RWAug_Search_H(ops_num, [0,0], [0,0], [0,0])
    
                rw_search.n = ops_num
                rw_search.group_ids = [trans_ids_lists[i][j][0] for j in range(ops_num)]
                rw_search.method_ids = [trans_ids_lists[i][j][1] for j in range(ops_num)]
                rw_search.strength_ids = [trans_ids_lists[i][j][2] for j in range(ops_num)]

                aug_prob = get_prob(aug_param_group,aug_param_methods,aug_param_strength,rw_search.group_ids,rw_search.method_ids,rw_search.strength_ids,ops_num)
                aug_probs.append(aug_prob)
                    
                logits,nowattack_mid = self.get_logits_features(net, self.transform(images+delta, search=rw_search))
                losses.append(self.get_loss(logits, label, math.floor((len(logits)+0.01)/len(label))).reshape(1))
                
                # 动态计算repeat_num，根据中间层输出的实际维度
                repeat_factor = math.floor((len(logits)+0.01)/len(label))
                repeat_num = (repeat_factor,) + (1,) * (len(mid_original.shape) - 1)
                mid_loss = Mid_layer_target_Loss()(mid_attack_original.repeat(repeat_num), nowattack_mid, mid_original.repeat(repeat_num))
                mid_losses.append(mid_loss.reshape(1))                   
                
            #计算在当前概率下的期望损失
            E_losses = torch.cat([aug_probs[i]*losses[i].reshape(1) for i in range(num_scale)])
            E_loss = torch.sum(E_losses)/num_scale


            #计算在当前概率下的期望中间层损失
            E_mid_losses = torch.cat([aug_probs[i]*mid_losses[i].reshape(1) for i in range(num_scale)])
            E_mid_loss = torch.sum(E_mid_losses)/num_scale

            if e !=0 :
                omega_C = E_loss/used_E_loss
                omega_F = E_mid_loss/used_E_mid_loss
                lamda = torch.exp(omega_F)/(torch.exp(omega_C)+torch.exp(omega_F))   
            used_E_loss = E_loss
            used_E_mid_loss = E_mid_loss

            lamda = torch.clamp(lamda, 0.1, 0.7)
            lamda = 0

            p_loss = E_loss + lamda*E_mid_loss


            # ------------------- PPO 更新 -------------------
            policy_epsilon = self.args.xi

            old_group_policy = aug_param_group.clone().detach()
            old_method_policy = aug_param_methods.clone().detach()
            old_strength_policy = aug_param_strength.clone().detach()

            for i in range(num_scale):
                PPO_optimizer.zero_grad()

                reward = losses[i] + lamda * mid_losses[i]

                advantage = aug_probs[i] * reward - p_loss
                advantage = advantage.detach()

                joint_old_prob = torch.tensor(1.0, device=device)
                joint_new_prob = torch.tensor(1.0, device=device)

                for j in range(ops_num):
                    gid = trans_ids_lists[i][j][0]
                    mid = trans_ids_lists[i][j][1]
                    sid = trans_ids_lists[i][j][2]
                    
                    p_g_old = old_group_policy[gid]
                    p_m_old = old_method_policy[gid, mid]
                    p_s_old = old_strength_policy[gid, mid, sid]

                    p_g_new = aug_param_group[gid]
                    p_m_new = aug_param_methods[gid, mid]
                    p_s_new = aug_param_strength[gid, mid, sid]

                    joint_old_prob *= (p_g_old * p_m_old * p_s_old)
                    joint_new_prob *= (p_g_new * p_m_new * p_s_new)

                ratio = (joint_new_prob + 1e-8) / (joint_old_prob + 1e-8)
                clipped = torch.clamp(ratio, 1 - policy_epsilon, 1 + policy_epsilon)

                ppo_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

                ppo_loss.backward()
                PPO_optimizer.step()

                
                with torch.no_grad():
                    aug_param_group.data.clamp_(min=0.75, max=1.25)
                    aug_param_methods.data.clamp_(min=0.75, max=1.25)
                    aug_param_strength.data.clamp_(min=0.75, max=1.25)
            
            
                
            #计算平均损失
            loss = torch.sum(torch.cat(losses))/num_scale
            #计算梯度用于更新扰动
            grad = self.get_grad(loss, delta)

            g = last_g * args.mu + grad / torch.norm(
                grad, p=1
            )
            used_delta = delta
            delta = delta + alpha * torch.sign(g)
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(delta, 0-images, 1-images)
            last_g = g

        return images+delta

    def attack(self):

        args = self.args
        device = args.device
        net = self.net.to(device)

        data_loader = DataLoader(
            self.dataset, batch_size=args.batch_size, num_workers=10,sampler=self.dataset_sampler
        )
        epsilon = args.eps / 255
        alpha = args.alpha / 255
        for images, te, filename in tqdm(data_loader):

            #print(filename)
            
            images_adv = self.l2t_forward_H(images.to(device), te.to(device))
            self.save_images(images_adv, te, filename)

        eval_result = self.eval()

        return eval_result
