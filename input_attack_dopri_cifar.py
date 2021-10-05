import time
import os
from tx2_predict import PowerLogger,getNodes
from PIL import Image
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchdiffeq import odeint_adjoint as odeint
from odeint import odeint
#from torchdiffeq import odeint
from torch import autograd
from torch.autograd import Variable
from torch import optim
import csv
#inter_op=torch.tensor([0, 1]).float()
#inter_op = autograd.Variable(torch.tensor([0, 1]).float(), requires_grad=True).cuda()
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEFullBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEFullBlock, self).__init__()
        self.conv1=nn.Conv2d(3, 64, 3, 1)
        self.norm1=norm(64)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(64, 64, 4, 2, 1)
        self.norm2=norm(64)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(64, 64, 4, 2, 1)
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        #norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)
        self.norm3=norm(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool1=nn.AdaptiveAvgPool2d((1, 1))
        self.flatten=Flatten()
        self.linear=nn.Linear(64, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.norm1(x)
        #iiop = torch.sum(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        #self.integration_time = self.integration_time.type_as(x)
        #self.integration_time = autograd.Variable(self.integration_time, requires_grad=True).cuda()
        out,iiop,n_steps = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3,method='dopri5')

        x=self.norm3(out[1])
        x=self.relu3(x)
        x=self.pool1(x)
        x=self.flatten(x)
        x=self.linear(x)
        '''print(out[1])
        print(iiop)
        time.sleep(5)'''
        #inter_op=iiop
        return x,iiop,n_steps

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value




class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        #self.integration_time = autograd.Variable(self.integration_time, requires_grad=True).cuda()
        out,iiop = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3,method='dopri5')
        '''print(out[1])
        print(iiop)
        time.sleep(5)'''
        inter_op=iiop
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut





def normalize(t):
    n = np.zeros(t.shape)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    t=t.astype('float')


    for i in range(1):
        for j in range(3):
            for k in range(32):
                for l in range(32):
                    t[i][j][k][l] = t[i][j][k][l] / 255.0
                    n[i][j][k][l]=(t[i][j][k][l]-mean[j]) /std[j]



    '''for i in range(3):
        t[ :, i, :, :] = t[ :, i, :, :] / 255.0
        n[ :, i, :, :] = (t[ :, i, :, :] - mean[i]) / std[i]
        # n[:, :, i, :, :] = n[:, :, i, :, :] / 255.0'''
    return n

def res(t):
    n = np.zeros((3,32,32))

    for j in range(3):
        for k in range(32):
            for l in range(32):
                n[j][k][l]=t[k][l][j]

    n=n.reshape(1,3,32,32)
    return n
def tanh_rescale(x, x_min=-1.7, x_max=2.05):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)

def loss_op(output, dist, scale_const):

    #loss1 =  target-output
    loss1 = scale_const * output
    loss2 = dist.sum()
    #print("loss1 ",loss1)
    #print("loss2 ",loss2)
    loss = loss1 + loss2
    return loss

    #return output

def create_image(py_list):
    #print("size ",py_list.size())
    p_list = py_list.tolist()[0]
    img = [[[0.0 for i1 in range(3)] for j1 in range(32)] for k1 in range(32)]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    #print("py_list ", py_list[0][0])
    for i1 in range(32):
        for j1 in range(32):
            p1 = ((p_list[0][i1][j1] * std[0] + mean[0]) * 255.0)
            p2 = ((p_list[1][i1][j1] * std[1] + mean[1]) * 255.0)
            p3 = ((p_list[2][i1][j1] * std[2] + mean[2]) * 255.0)
            l = [p1, p2, p3]

            img[i1][j1] = l

    img_arr = np.asarray(img)
    #print("img_arr ",img_arr[0][0])
    img_arr = np.rint(img_arr)
    img_arr = img_arr.astype(np.uint8)
    Img = Image.fromarray(img_arr, 'RGB')
    img_arr2=np.array(Img)
    #print("img_arr ", img_arr[0][0])

    #img_arr2 = np.asarray(img_arr)
    img_arr = res(img_arr2)
    img_arr=img_arr.astype('float')
    img_arr = normalize(img_arr)
    return img_arr,img_arr2


'''def create_image(py_list,folder,index,steps,max_ind):
    #print("size ",py_list.size())
    p_list = py_list.tolist()[0]
    img = [[[0.0 for i1 in range(3)] for j1 in range(32)] for k1 in range(32)]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    for i1 in range(32):
        for j1 in range(32):
            p1 = int((p_list[0][i1][j1] * std[0] + mean[0]) * 255)
            p2 = int((p_list[1][i1][j1] * std[1] + mean[1]) * 255)
            p3 = int((p_list[2][i1][j1] * std[2] + mean[2]) * 255)
            l = [p1, p2, p3]

            img[i1][j1] = l

    img_arr = np.asarray(img)
    new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')
    #new_im.save("output/img_hello.png")
    new_im.save(folder+"/image_" + str(index) +"_"+str(steps)+"_"+str(max_ind)+ ".png")'''



device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
downsampling_layers = [
        nn.Conv2d(3, 64, 3, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
    ]

is_odenet =  'odenet'


transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

testset = datasets.__dict__['cifar10'.upper()](root='data/',
                                                             train=True,
                                                             download=True,
                                                             transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)
feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

#model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
model=ODEFullBlock(ODEfunc(64)).to(device)
'''test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=1, shuffle=False, num_workers=2, drop_last=True
    )'''

checkpoint = torch.load('model_cifar2.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval().cuda()
criterion = nn.CrossEntropyLoss().to(device)
index=0
orig=[]
advs=[]
c_vals=[10,100,1000]
for x, y in test_loader:
    if (index >= 100):
        break
    print("in")
    input_var = Variable(x, volatile=True).cuda()

    orig_np = input_var.data.cpu().numpy()
    orig.append(orig_np.tolist())
    np.save(
        "dopri_cifar_org.npy", np.asarray(orig))
    logits, iiop, n_steps = model(input_var)
    orig_n=n_steps
    #time.sleep(5)
    y=y.to(device)
    #logits = model(input_var)
    #time.sleep(2)
    #print("end")
    modifier = torch.rand(input_var.size(), device="cuda").float()
    new_input_adv = torch.zeros(input_var.size()).float()
    modifier_var = autograd.Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier_var], lr=0.0005)
    target = torch.tensor(5 * [1.0], device="cuda")
    # target_var = autograd.Variable(target, requires_grad=False)
    min_loss = float("inf")
    # adv_img_min = np.zeros((1, 32, 32, 3))
    # min_output = torch.tensor(0.0)
    max_n = 0
    max_ind=0
    for c_ind in range(len(c_vals)):
        c=c_vals[c_ind]
        for ind in range(2000):
            # print(ind)
            scale_const_var = torch.tensor(c)
            scale_const_var = autograd.Variable(scale_const_var, requires_grad=False).cuda()
            # print(modifier_var)
            input_adv = tanh_rescale(modifier_var + input_var, -1.7, 2.05)
            logits, iiop, n_steps = model(input_adv)
            dist = l2_dist(input_adv, input_var, keepdim=False)
            loss = loss_op(iiop, dist, scale_const_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info = []
            info.append(n_steps)
            adv_np = input_adv.data.cpu().numpy()
            # mod_np = modifier_var.data.cpu().numpy()
            adv_np, img_arr2 = create_image(adv_np)

            input_adv2 = torch.from_numpy(adv_np)

            input_adv2 = input_adv2.type(torch.cuda.FloatTensor)
            input_adv2 = input_adv2.to(device)
            logits, iiop, n_steps = model(input_adv2)

            info.append(n_steps)

            if (max_n < n_steps):
                # temp=sm2
                max_ind = ind
                min_loss = loss
                max_n = n_steps
                # adv_img_min = input_adv_np
                new_input_adv = input_adv2
            # print("end")
        # time.sleep(10)
        info = []
        info.append(c)
        info.append(orig_n)
        info.append(max_n)
        f = open("dopri_results_cifar.csv", "a")
        writer = csv.writer(f)
        # print(energy)
        writer.writerow(info)
        f.close()
        if (index < 500):
            adv_np = new_input_adv.data.cpu().numpy()
            advs.append(adv_np.tolist())
            np.save(
                "dopri_cifar_adv.npy", np.asarray(advs))
            # create_image(new_input_adv, 'adverse_dopri_single2', index, max_n,max_ind)
    index += 1




