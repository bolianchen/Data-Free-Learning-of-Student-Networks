#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import tqdm
import numpy as np
import math
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models

from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from models import Generator
from utils import kdloss, adjust_learning_rate, AvgrageMeter, accuracy

import resnet
from lenet import LeNet5
from lenet import LeNet5Half


def train_teacher(teacher, data_train_loader, data_test_loader, optimizer,
                  num_epochs):
    """ train a teacher model on a specified dataset
    """
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(num_epochs):
        # train
        teacher.train()
        for i, (images, labels) in enumerate(data_train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = teacher(images)
            loss = criterion(output, labels)

            loss.backward()
            prec, = accuracy(output, labels)
            optimizer.step()
            n = images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec.item(), n)

            if i % 50 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {i*50}; '\
                      f'loss = {objs.avg}, acc = {top1.avg}')
        # test
        objs.reset()
        top1.reset()
        teacher.eval()

        with torch.no_grad():
            for images_test, labels_test in data_test_loader:
                images_test, labels_test = images_test.cuda(
                ), labels_test.cuda()
                output_test = teacher(images_test)
                loss_test = criterion(output_test, labels_test)
                prec_test, = accuracy(output_test, labels_test)

                n_test = images_test.size(0)
                objs.update(loss_test.item(), n_test)
                top1.update(prec_test.item(), n_test)

        print(f'Epoch {epoch}/{num_epochs}; Test Acc = {top1.avg}')


def test(model, data_test_loader):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    model.eval()
    with torch.no_grad():
        for i, (images_test, labels_test) in enumerate(data_test_loader):
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            output_test = model(images_test)
            loss_test = criterion(output_test, labels_test)
            prec_test, = accuracy(output_test, labels_test)

            n_test = images_test.size(0)
            objs.update(loss_test.item(), n_test)
            top1.update(prec_test.item(), n_test)
            if i % 50 == 0:
                print(f'Finished {i+1}/{len(data_test_loader)}')

    print(f'Avg Loss = {objs.avg}' f'Test Acc = {top1.avg}')


def main(opt):
    """
    """
    print(f'image shape: {opt.channels} x {opt.img_size} x {opt.img_size}')

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    accr = 0
    accr_best = 0

    generator = Generator(opt).to(device)

    if opt.dataset == 'imagenet':
        assert opt.teacher_model_name != 'none', 'DAFL does not support imagene'
        teacher = eval(f'models.{opt.teacher_model_name}(pretrained = True)')
        teacher = teacher.to(device)
        # teacher.eval()
        assert opt.student_model_name != 'none', 'DAFL does not support imagenet'
        net = eval(f'models.{opt.student_model_name}(pretrained = False)')
        net = net.to(device)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # for optimizing the teacher model
        if opt.train_teacher:
            data_train = torchvision.datasets.ImageNet(
                opt.data_dir, split='train', transform=transform_train)
            data_train_loader = DataLoader(data_train,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)
            optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)

        # for optimizing the student model
        data_test = torchvision.datasets.ImageNet(opt.data_dir,
                                                  split='val',
                                                  transform=transform_test)
        data_test_loader = DataLoader(data_test,
                                      batch_size=opt.batch_size,
                                      num_workers=4,
                                      shuffle=False)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
        optimizer_S = torch.optim.SGD(net.parameters(),
                                      lr=opt.lr_S,
                                      momentum=0.9,
                                      weight_decay=5e-4)

    else:
        if opt.dataset == 'MNIST':
            # use the original DAFL network
            if opt.teacher_model_name == 'none':
                teacher = LeNet5().to(device)
            # use torchvision models
            else:
                teacher = eval(
                    f'models.{opt.teacher_model_name}(pretrained = False)')
                teacher.conv1 = nn.Conv2d(
                    1, teacher.conv1.out_channels, teacher.conv1.kernel_size,
                    teacher.conv1.stride, teacher.conv1.padding,
                    teacher.conv1.dilation, teacher.conv1.groups,
                    teacher.conv1.bias, teacher.conv1.padding_mode)
                teacher.fc = nn.Linear(teacher.fc.in_features, 10)
                teacher = teacher.to(device)

            # use the original DAFL network
            if opt.student_model_name == 'none':
                net = LeNet5Half().to(device)
            # use torchvision models
            else:
                net = eval(f'models.{opt.student_model_name}()')
                net.conv1 = nn.Conv2d(1, net.conv1.out_channels,
                                      net.conv1.kernel_size, net.conv1.stride,
                                      net.conv1.padding, net.conv1.dilation,
                                      net.conv1.groups, net.conv1.bias,
                                      net.conv1.padding_mode)
                net.fc = nn.Linear(net.fc.in_features, 10)
                net = net.to(device)

            # for optimizing the teacher model
            if opt.train_teacher:
                data_train = MNIST(opt.data_dir,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307, ),
                                                            (0.3081, ))
                                   ]))
                data_train_loader = DataLoader(data_train,
                                               batch_size=256,
                                               shuffle=True,
                                               num_workers=4)
                optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)

            # for optimizing the student model
            data_test = MNIST(opt.data_dir,
                              download=True,
                              train=False,
                              transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307, ), (0.3081, ))
                              ]))
            data_test_loader = DataLoader(data_test,
                                          batch_size=64,
                                          num_workers=4,
                                          shuffle=False)
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
            optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

        elif opt.dataset == 'cifar10':
            # use the original DAFL network
            if opt.teacher_model_name == 'none':
                teacher = resnet.ResNet34().to(device)
            # use torchvision models
            else:
                teacher = eval(
                    f'models.{opt.teacher_model_name}(pretrained = False)')
                teacher.fc = nn.Linear(teacher.fc.in_features, 10)
                teacher = teacher.to(device)

            # use the original DAFL network
            if opt.student_model_name == 'none':
                net = resnet.ResNet18().to(device)

            # use torchvision models
            else:
                net = eval(f'models.{opt.student_model_name}()')
                net.fc = nn.Linear(net.fc.in_features, 10)
                net = net.to(device)

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # for optimizing the teacher model
            if opt.train_teacher:
                data_train = CIFAR10(opt.data_dir,
                                     download=True,
                                     transform=transform_train)
                data_train_loader = DataLoader(data_train,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4)
                optimizer = torch.optim.SGD(teacher.parameters(),
                                            lr=0.1,
                                            momentum=0.9,
                                            weight_decay=5e-4)

            # for optimizing the student model
            data_test = CIFAR10(opt.data_dir,
                                download=True,
                                train=False,
                                transform=transform_test)
            data_test_loader = DataLoader(data_test,
                                          batch_size=100,
                                          num_workers=4)
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
            optimizer_S = torch.optim.SGD(net.parameters(),
                                          lr=opt.lr_S,
                                          momentum=0.9,
                                          weight_decay=5e-4)

        elif opt.dataset == 'cifar100':
            # use the original DAFL network
            if opt.teacher_model_name == 'none':
                teacher = resnet.ResNet34(num_classes=100).to(device)
            # use torchvision models
            else:
                teacher = eval(
                    f'models.{opt.teacher_model_name}(pretrained = False)')
                teacher.fc = nn.Linear(teacher.fc.in_features, 100)
                teacher = teacher.to(device)

            # use the original DAFL network
            if opt.student_model_name == 'none':
                net = resnet.ResNet18(num_classes=100).to(device)
            # use torchvision models
            else:
                net = eval(f'models.{opt.student_model_name}()')
                net.fc = nn.Linear(net.fc.in_features, 100)
                net = net.to(device)

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

            # for optimizing the teacher model
            if opt.train_teacher:
                data_train = CIFAR100(opt.data_dir,
                                      download=True,
                                      transform=transform_train)
                data_train_loader = DataLoader(data_train,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4)
                optimizer = torch.optim.SGD(teacher.parameters(),
                                            lr=0.1,
                                            momentum=0.9,
                                            weight_decay=5e-4)

            # for optimizing the student model
            data_test = CIFAR100(opt.data_dir,
                                 download=True,
                                 train=False,
                                 transform=transform_test)
            data_test_loader = DataLoader(data_test,
                                          batch_size=100,
                                          num_workers=4)
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
            optimizer_S = torch.optim.SGD(net.parameters(),
                                          lr=opt.lr_S,
                                          momentum=0.9,
                                          weight_decay=5e-4)

    # train the teacher model on the specified dataset
    if opt.train_teacher:
        train_teacher(teacher, data_train_loader, data_test_loader, optimizer,
                      opt.n_epochs_teacher)

    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)
        generator = nn.DataParallel(generator)
        net = nn.DataParallel(net)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    if opt.pretest:
        test(teacher, data_test_loader)

    # ----------
    #  Training
    # ----------
    batches_done = 0
    for epoch in range(opt.n_epochs):
        total_correct = 0
        avg_loss = 0.0
        if opt.dataset != 'MNIST':
            adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

        for i in range(120):
            net.train()
            z = torch.randn(opt.batch_size, opt.latent_dim).cuda()
            optimizer_G.zero_grad()
            optimizer_S.zero_grad()
            gen_imgs = generator(z)
            # teacher inference should not calculate gradients
            if opt.dataset != 'imagenet' and opt.teacher_model_name == 'none':
                outputs_T, features_T = teacher(gen_imgs, out_feature=True)
            else:
                features = [torch.Tensor().cuda(0)]

                def hook_features(model, input, output):
                    features[0] = torch.cat((features[0], output.cuda(0)), 0)

                if torch.cuda.device_count() > 1:
                    teacher.module.avgpool.register_forward_hook(hook_features)
                else:
                    teacher.avgpool.register_forward_hook(hook_features)
                outputs_T = teacher(gen_imgs)
                features_T = features[0]

            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T, pred)
            softmax_o_T = torch.nn.functional.softmax(outputs_T,
                                                      dim=1).mean(dim=0)
            loss_information_entropy = (softmax_o_T *
                                        torch.log10(softmax_o_T)).sum()
            loss = (loss_one_hot * opt.oh + loss_information_entropy * opt.ie +
                    loss_activation * opt.a)

            loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())

            loss += loss_kd

            loss.backward()
            optimizer_G.step()
            optimizer_S.step()
            if i == 1:
                print( f'[Epoch {epoch}/{opt.n_epochs}]'\
                         '[loss_oh: {loss_one_hot.item()}]'\
                         '[loss_ie: {loss_information_entropy.item()}]'\
                         '[loss_a: {loss_activation.item()}]'\
                         '[loss_kd: {loss_kd.item()}]' )

        test(net, data_test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='MNIST',
                        choices=['MNIST', 'cifar10', 'cifar100', 'imagenet'],
                        help='path to the dataset folder')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./',
                        help='path to the dataset folder')

    parser.add_argument('--train_teacher',
                        action='store_true',
                        help='whether to train the teacher model from scratch')
    parser.add_argument('--pretest',
                        action='store_true',
                        help='whether to test the teacher model'
                        ' before training the student model')
    parser.add_argument('--teacher_model_name',
                        type=str,
                        default='wide_resnet50_2',
                        choices=[
                            'none', 'resnet18', 'inception_v3', 'googlenet',
                            'inception_v3', 'wide_resnet50_2', 'mnasnet1_0'
                        ],
                        help='all the torchvision models are applicable'
                        ' please check https://pytorch.org/docs/stable/'
                        'torchvision/models.html')
    parser.add_argument('--student_model_name',
                        type=str,
                        default='resnet18',
                        choices=[
                            'none', 'resnet18', 'inception_v3', 'googlenet',
                            'inception_v3', 'wide_resnet50_2', 'mnasnet1_0'
                        ],
                        help='all the torchvision models are applicable'
                        ' please check https://pytorch.org/docs/stable/'
                        'torchvision/models.html')
    parser.add_argument(
        '--teacher_dir',
        type=str,
        default='./teachers',
        help='path to the folder of the teacher model checkpoint')
    parser.add_argument('--n_epochs_teacher',
                        type=int,
                        default=200,
                        help='number of epochs to train teachers')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=200,
                        help='number of epochs to train students')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr_G',
                        type=float,
                        default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_S',
                        type=float,
                        default=2e-3,
                        help='learning rate')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--img_size',
                        type=int,
                        default=32,
                        help='size of each image dimension')
    parser.add_argument('--channels',
                        type=int,
                        default=3,
                        help='number of image channels')
    parser.add_argument('--oh', type=float, default=1, help='one hot loss')
    parser.add_argument('--ie',
                        type=float,
                        default=5,
                        help='information entropy loss, urge the generator to'
                        ' produce data with balanced classes')
    parser.add_argument(
        '--a',
        type=float,
        default=0.1,
        help='activation loss, the absolute value of activation'
        ' right before the fully connected layer')
    parser.add_argument('--output_dir', type=str, default='./')
    opt = parser.parse_args()

    main(opt)
