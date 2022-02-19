import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD
from networks.resnet_big import SupConResNet, LinearClassifier
import torch.backends.cudnn as cudnn
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import sys
import argparse
import time
import math


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # parser.add_argument('--print_freq', type=int, default=10,
    #                     help='print frequency')
    # parser.add_argument('--save_freq', type=int, default=50,
    #                     help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    # parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
    #                     help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.2,
    #                     help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum')

    # model dataset
    parser.add_argument('--teacher_model', type=str, default='resnet50')
    parser.add_argument('--student_model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    # parser.add_argument('--warm', action='store_true',
    #                     help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.dataset != 'path':
        opt.data_folder = './datasets/'
    
    

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.student_model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    # if opt.cosine:
    #     opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     if opt.cosine:
    #         eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #         opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #                 1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    #     else:
    #         opt.warmup_to = opt.learning_rate

    if opt.dataset in ('cifar10', 'path'):
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt




def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = (0.41044191,0.45704237,0.46365224)
        std = std = (4.37454361,4.06389989,4.11659655)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder+'train/',
                                            transform=train_transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder+'test/',
                                            transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, #(train_sampler is None),
        num_workers=opt.num_workers, #pin_memory=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader




def set_teacher_model(opt):
    model = SupConResNet(name=opt.teacher_model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.teacher_model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion



def set_student_model(opt):
    model = SupConResNet(name=opt.student_model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion





opt = parse_option()

train_loader, test_loader = set_loader(opt)

teacher_model, classifier, te_criterion = set_teacher_model(opt)
student_model, st_criterion = set_student_model(opt)

teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
student_optimizer = optim.SGD(student_model.parameters(), 0.01)

distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                      teacher_optimizer, student_optimizer)
# # distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)    # Train the teacher network
# distiller.train_student(epochs=5, plot_losses=True, save_model=True)    # Train the student network
# distiller.evaluate(teacher=False)                                       # Evaluate the student network
# distiller.get_parameters()                                              # A utility function to get the number of 
                                                                        # parameters in the  teacher and the student network
                                                                        