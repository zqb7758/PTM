"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI, gkern, Admix, PT
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from styleaug import StyleAugmentor
from torch import nn


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=5, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations (Sampling Number)")
parser.add_argument("--portion", type=float, default=0.2, help="protion for the mixed image")
parser.add_argument("--gamma", type=float, default=0.5, help="protion for the mixed original image")
parser.add_argument("--beta", type=float, default=2.0, help="weighted")
parser.add_argument("--attack_method", type=str, default="PTM", choices=["STM", "MIFGSM", "NIFGSM", "DIM", "TIM", "PTM", "SIM", "Admix_MI"], help="Attack method to use")
parser.add_argument("--model", type=str, default="inceptionv3", choices=["inceptionv3", "inceptionv4", "resnet101", "inceptionresnetv2"], help="Target model for generating adversarial examples")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

T_kernel = gkern(7, 3)


def STM(images, gt, model, min, max): 

    Resize = T.Resize(size=(299, 299))
    momentum = opt.momentum
    num_iter = opt.num_iter_set
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = 0
    N = opt.N
    beta = opt.beta
    gamma = opt.gamma
    augmentor = StyleAugmentor()
    for i in range(num_iter):
        noise = 0
        for n in range(N):
            x_aug = augmentor(x)
            x_new = gamma*x + (1-gamma)*Resize(x_aug.detach()).clone() + torch.randn_like(x).uniform_(-eps*beta, eps*beta)
            x_new = V(x_new, requires_grad = True)
            
            output_v3 = model(x_new)
            
            loss = F.cross_entropy(output_v3, gt)
            
            noise += torch.autograd.grad(loss, x_new,
                                        retain_graph=False, create_graph=False)[0]
        noise = noise / N

        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def MIFGSM(images, gt, model, min, max):
    """
    The attack algorithm of MI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / 10.0
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)

        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def NIFGSM(images, gt, model, min, max):
    """
    The attack algorithm of NI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)
        nes_x = x + momentum * alpha * grad
        output_v3 = model(nes_x)
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def DIM(images, gt, model, min, max):
    """
    The attack algorithm of DIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)
        output_v3 = model(DI(x))
        if isinstance(output_v3, list):
            output_v3 = output_v3[0]
        else:
            output_v3 = output_v3
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def TIM(images, gt, model, min, max):
    """
    The attack algorithm of TIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    T_kernel = gkern(7, 3)
    for i in range(num_iter):
        x = V(x, requires_grad = True)
        
        output_v3 = model(x)
        
        loss = F.cross_entropy(output_v3, gt)
        
        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def PTM(images, gt, model, mini, maxi):
    """
    The attack algorithm of RTM using random transformations
    :param x: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    if 'resnet' in model:
        Resize = T.Resize(size=(224,224))
    else:
        Resize = T.Resize(size=(299,299))
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    if isinstance(num_iter, torch.Tensor):
        num_iter = num_iter.item()  
    num_iter = int(num_iter)
    alpha = eps / 10.0
    momentum = opt.momentum
    beta = opt.beta
    gamma = opt.gamma
    N = opt.N
    tensor_values = torch.arange(0, N)
    tilt_values = tensor_values.tolist()
    x = images.clone()
    for i in range(num_iter):
        noise = 0
        for tilt in tilt_values:
            
            transformed_x, angle= PT(x, tilt=tilt)
            x_new = gamma*x + (1-gamma)*Resize(transformed_x.detach()).clone() + torch.randn_like(x).uniform_(-eps*beta, eps*beta)
            x_new= V(x_new, requires_grad=True)

            output_v3 = model(x_new)
            
            loss = F.cross_entropy(output_v3, gt)

            noise += torch.autograd.grad(loss, x_new,
                                    retain_graph=False, create_graph=False)[0]

        noise = noise / N
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, mini, maxi)
    return x.detach()

def SIM(images, gt, model, min, max):
    """
    The attack algorithm of SIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    for i in range(num_iter):
        x = V(x, requires_grad = True)
        noise = torch.zeros_like(x).detach().cuda()
        for i in torch.arange(5):
            nes_x = x / torch.pow(2, i)
            
            output_v3 = model(nes_x)
            
            loss = F.cross_entropy(output_v3, gt)
            
            noise += torch.autograd.grad(loss, x,
                                        retain_graph=False, create_graph=False)[0]
        noise = noise / 5
        
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def Admix_MI(images, gt, model, min, max):
    """
    The attack algorithm of Admix
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    portion = opt.portion
    size = 3
    x = images.clone().detach().cuda()

    g_t = torch.cat([torch.cat([(gt) for _ in range(size)]) for _ in range(5)]) 
    for i in range(num_iter):
        x = V(x, requires_grad = True)
        admix = Admix(x, size, portion)
        x_batch = torch.cat([admix, admix/2, admix/4, admix/8, admix/16], axis=0)

        output_v3 = model(x_batch)
        
        loss = F.cross_entropy(output_v3, g_t)

        noise_total = torch.autograd.grad(loss, x_batch,
                                    retain_graph=False, create_graph=False)[0]
        noise1, noise2, noise3, noise4, noise5 = torch.split(noise_total, x.shape[0]*size, dim=0)
        avg1_noise = 1/5*(noise1+noise2/2+noise3/4+noise4/8+noise5/16)
        avg1, avg2, avg3 = torch.split(avg1_noise, x.shape[0], dim=0)
        noise = 1/3*(avg1+avg2+avg3)
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def get_model(model_name):

    model_params = {
        "inceptionv3": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.5, 0.5, 0.5]),
            "input_size": (299, 299)
        },
        "inceptionv4": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.5, 0.5, 0.5]),
            "input_size": (299, 299)
        },
        "resnet101": {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
            "input_size": (224, 224)
        },
        "resnet152": {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
            "input_size": (224, 224)
        },
        "inceptionresnetv2": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.5, 0.5, 0.5]),
            "input_size": (299, 299)
        }
    }
    
    # Get parameters for the selected model
    if model_name in ["inceptionresnetv2,inceptionv3,inceptionv4","inceptionv3", "inceptionv4", "resnet50", "resnet101", "resnet152", "densenet121", "vgg16","inceptionresnetv2"]:
        params = model_params.get(model_name)

    model_path = os.path.join('./models', model_name + '.npy')
    # Create the model based on the selection
    if model_name == "inceptionv3":
        model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet')
    elif model_name == "inceptionv4":
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
    elif model_name == "resnet101":
        model = pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet')
    elif model_name == "resnet152":
        model = pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')
    elif model_name == "inceptionresnetv2":
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    elif model_name == 'tf2torch_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        return nn.Sequential( 
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),),(299, 299)
    elif model_name == 'tf2torch_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        return nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),),(299, 299)
    elif model_name == 'tf2torch_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        return nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),),(299, 299)
    elif model_name == 'tf2torch_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        return nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),),(299, 299)
    else:
        raise ValueError(f"Unsuported Model: {model_name}")
    
    # Return the model with normalization layer
    return torch.nn.Sequential(
        Normalize(params["mean"], params["std"]),
        model.eval().cuda()
    ), params["input_size"]

def main():

    output_dir = f'./outputs/{opt.model}_{opt.attack_method.lower()}_outputs/'

    attack_methods = {
        "STM": STM,
        "MIFGSM": MIFGSM,
        "NIFGSM": NIFGSM,
        "DIM": DIM,
        "TIM": TIM,
        "PTM": PTM,
        "SIM": SIM,
        "Admix_MI": Admix_MI
    }
    
    # Create model based on the command-line argument
    model, input_size = get_model(opt.model)
    # Update image dimensions based on the model
    X = ImageNet(opt.input_dir, opt.input_csv, 
                T.Compose([T.Resize(input_size), T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # ==================================================================
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    for images, images_ID, gt_cpu in tqdm(data_loader):

        gt = gt_cpu.cuda()

        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        
        attack_func = attack_methods[opt.attack_method]
        
        adv_img5 = attack_func(images, gt, model, images_min, images_max)
        adv_img5_cpu = adv_img5.cpu()
        
        del adv_img5
        torch.cuda.empty_cache()
        
        adv_img_np5 = adv_img5_cpu.numpy()
        adv_img_np5 = np.transpose(adv_img_np5, (0, 2, 3, 1)) * 255
        if attack_func == PTM:
            adv_img_np5 = np.transpose(adv_img_np5, (0, 2, 1, 3))
        save_image(adv_img_np5, images_ID, output_dir)
        
        del adv_img5_cpu, adv_img_np5
        del images, gt, images_min, images_max
        torch.cuda.empty_cache()
