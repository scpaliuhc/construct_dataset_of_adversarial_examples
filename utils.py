import os 
import PIL.Image as Image
import torch
from torchvision import transforms
from torchvision.models.resnet import resnet101
import torchvision.models as models
from foolbox import PyTorchModel
import torch.nn as nn

def load_images(root='/data0/lhc/dataset/IQA_AE/REF/REF_VOC',w=512,model=None):
    tran=transforms.Compose([
            transforms.Resize((w,w)),
            transforms.ToTensor()])
    files_=[]
    images_=[]
    labels_=[]
    for cls in os.listdir(root):
        for file in os.listdir(os.path.join(root,cls)):
            path = os.path.join(root,cls,file)
            image = Image.open(path)
            image = tran(image).unsqueeze(0)
            images_.append(image)
            files_.append(file)
            if model is not None:
                pred=model(image.to('cuda'))
                label=pred.argmax().cpu().item()
                labels_.append(label)
                # print(path,label)
    images_=torch.cat(images_,dim=0)
    if model is not None:
        labels_=torch.tensor(labels_)
    return images_,labels_,files_

def load_target_model(type):
    if type=='foolbox':
        print('load model (foolbox)')
        model = models.inception_v3(pretrained=True).eval()
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        model = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    elif type=='torchattacks':
        print('load model (torchattacks)')
        class Normalize(nn.Module) :
            def __init__(self, mean, std) :
                super(Normalize, self).__init__()
                self.register_buffer('mean', torch.Tensor(mean))
                self.register_buffer('std', torch.Tensor(std))
            def forward(self, input):
                # Broadcasting
                mean = self.mean.reshape(1, 3, 1, 1)
                std = self.std.reshape(1, 3, 1, 1)
                return (input - mean) / std
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(
            norm_layer,
            models.inception_v3(pretrained=True)
        )
        model=model.eval()
        model.cuda()
    return model

##Simba
import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct

# mean and std for different datasets
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_TRANSFORM = trans.Compose([
    trans.Resize(342),
    trans.CenterCrop(299),
    trans.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = trans.Compose([
    trans.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = trans.Compose([
    trans.ToTensor()])


# reverses the normalization transformation
def invert_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    imgs_trans = imgs.clone()
    if len(imgs.size()) == 3:
        for i in range(imgs.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(imgs.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans


# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


# get most likely predictions and probabilities for a set of inputs
def get_preds(model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# get least likely predictions and probabilities for a set of inputs
def get_least_likely(model, inputs, dataset_name, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        prob, pred = output.min(1)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# defines a diagonal order
# order is fixed across diagonals but are randomized across channels and within the diagonal
# e.g.
# [1, 2, 5]
# [3, 4, 8]
# [6, 7, 9]
def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, :(image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(0, torch.LongTensor([i for i in range(i-1, -1, -1)]))
        order[i, (image_size - i):] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]


# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]
def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# zeros all elements outside of the top-left (block_size * ratio) submatrix for every block
def block_zero(x, block_size=8, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = torch.zeros(x.size(0), x.size(1), block_size, block_size)
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] * mask
    return z


# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            submat_dct = dct(dct(submat, axis=2, norm='ortho'), axis=3, norm='ortho')
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z
##Simba


import torchvision.datasets as dsets
def image_folder_custom_label(root, transform, idx2label) :
    
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data

if __name__=='__main__':
    # import json
    # class_idx=json.load(open("./imagenet_class_index.json"))
    # idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    # print(idx2label)
    # # transform = transforms.Compose([
    # #     transforms.Resize((299, 299)),
    # #     transforms.ToTensor(),])
    # # imagnet_data = image_folder_custom_label(root='/data1/public_dataset/Image__ILSVRC2012/ILSVRC2012_img_train/train', transform=transform, idx2label=idx2label)
    # # data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=2, shuffle=False)
    # # images, labels = iter(data_loader).next()
    # # print(labels)
    
    
    iamges,size,files=load_images()
