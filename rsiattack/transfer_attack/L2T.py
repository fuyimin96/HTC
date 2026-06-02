import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsiattack import ATTACK
from .utils.L2T_transform import op_list, RWAug_Search, select_op, trace_prob
import math


class L2T(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)

        self.random_start = True
        self.norm = 'linfty'

        super(L2T, self).__init__(self.args)

    def add_options(self, parser):
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--eps", type=float, default=16)
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--mu", type=float, default=1.0)
        parser.add_argument("--num_scale", type=int, default=8)
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def get_logits(self, x, **kwargs):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.net(x)
    
    def get_loss(self, logits, label, num_copy):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return self.loss(logits, label.repeat(num_copy))

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
    
    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.args.mu + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def transform(self, x, **kwargs):
        return kwargs['search'](x)
    
    def init_delta(self, data, **kwargs):
        args = self.args
        epsilon = args.eps / 255

        delta = torch.zeros_like(data).to(self.args.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-epsilon, epsilon)
            else:
                delta.normal_(-epsilon, epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*epsilon
            delta = torch.clamp(delta, 0.0-data, 1.0-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        args = self.args
        epsilon = args.eps / 255

        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta)
        delta = torch.clamp(delta, 0.0-data, 1.0-data)
        return delta
    
    def l2t_forward(self, images, label, **kwargs):
            
            args = self.args
            device = args.device
            #net = self.net.to(device)
            net = self.net.to(device)
            aug_length = len(op_list)
            ops_num = 2
            learning_rate = 0.01
            #self.num_scale = 10
            epsilon = args.eps / 255
            alpha = args.alpha / 255

            aug_param = torch.nn.Parameter(torch.rand(aug_length,requires_grad=True)*10,requires_grad=True)      

            images = images.clone().detach().to(device)
            label = label.clone().detach().to(device)

            # Initialize adversarial perturbation
            delta = self.init_delta(images)
            

            momentum = 0
            for e in range(args.epochs):
                # transform data
                aug_probs = []
                losses = []
                
                for i in range(args.num_scale):
                    rw_search = RWAug_Search(ops_num, [0,0])
                    
                    augtype = (ops_num, select_op(aug_param, ops_num))
                    aug_prob = trace_prob(aug_param, augtype[1])

                    rw_search.n = augtype[0]
                    rw_search.idxs = augtype[1]
                    
                    aug_probs.append(aug_prob)
                    
                    logits = net(self.transform(images+delta, search=rw_search))
                    
                    losses.append(self.get_loss(logits, label, math.floor((len(logits)+0.01)/len(label))).reshape(1))

                # Calculate the loss
                loss = torch.sum(torch.cat(losses))/args.num_scale
                
                # Calculate the gradients
                grad = self.get_grad(loss, delta)
                
                aug_losses = torch.cat([aug_probs[i]*losses[i].reshape(1) for i in range(args.num_scale)])
                aug_loss = torch.sum(aug_losses)/args.num_scale

                aug_grad = torch.autograd.grad(aug_loss, aug_param, retain_graph=False, create_graph=False)[0]
                aug_param = aug_param + learning_rate * aug_grad
                

                momentum = self.get_momentum(grad, momentum)

                delta = self.update_delta(delta, images, momentum, alpha)
                torch.cuda.empty_cache()

            return images + delta

    def attack(self):

        args = self.args
        device = args.device
        net = self.net.to(device)

        data_loader = DataLoader(
            self.dataset, batch_size=args.batch_size, num_workers=10
        )

        for images, te, filename in tqdm(data_loader):
            
            images_adv = self.l2t_forward(images.to(device), te.to(device))
            self.save_images(images_adv, te, filename)

        eval_result = self.eval()

        return eval_result

