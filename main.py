#	author: Yiying WEI & Guochu YI
#  	date: 2021/7/1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import time
import moxing
import argparse
#from config import get_args
from ntu_rgb_preprocess import load_data
from model import Model
model_dir = "/cache/result"
data_dir = "./dataset"
checkpoints_dir = "/cache/checkpoints"

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters




def save_checkpoint(state, is_best):
    torch.save(state, os.path.join(model_dir, 'Model-Best-NEW.pth.tar'))
        
    if is_best:
        print('Best Model Saving ...')
        moxing.file.copy_parallel(model_dir, args.train_url)


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    train_acc = 0.0
    step = 0
    for x1, x2, x3, x4, x5, traversal, target in train_loader:
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            x1, x2, x3, x4, x5, traversal, target =\
                x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(),\
                traversal.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(x1, x2, x3, x4, x5, traversal)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]
        acc = float(y_pred.eq(target.data).sum()) / len(x1) * 100.
        train_acc += acc
        step += 1
        #if step % args.print_interval == 0:
        if step % 10 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%, time interval: {3}".format(epoch, loss.data, acc, time.time() - start_time), end=' ')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))


def eval(model, test_loader, args):
    print('evaluation ...')
    model.eval()
    correct = 0
    with torch.no_grad():
        for x1, x2, x3, x4, x5, traversal, target in test_loader:
            if args.cuda:
                x1, x2, x3, x4, x5, traversal, target = \
                    x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(), \
                    traversal.cuda(), target.cuda()
            output = model(x1, x2, x3, x4, x5, traversal)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}%'.format(acc))
    return acc


def main(args):
    # 1. load data
    train_loader, test_loader = load_data(args)

    # 2. define model
    model = Model(temporal_size=args.temporal_size, joint_sequence_size=117, num_classes=60)
    if args.pretrained:
        # file_path = './model_best.pth.tar'
        # checkpoint = torch.load(file_path)
        os.makedirs(checkpoints_dir)
        moxing.file.copy_parallel(args.train_url, checkpoints_dir)
        checkpoint = torch.load(os.path.join(checkpoints_dir, 'Model-Best.pth.tar'))
        best_acc = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print('pretrained model loading ...')
        print('best acc : {}, start epoch : {}'.format(best_acc, start_epoch))
    else:
        best_acc = 0.0
        start_epoch = 1

    if args.cuda:
        model = model.cuda()
    print('model parameters :: {}'.format(get_model_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, args)
        eval_acc = eval(model, test_loader, args)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
            'parameters': get_model_parameters(model)
        }, is_best)
         

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser('two stream recurrent neural network')
    parser.add_argument('--data_url', type=str, default='./dataset')
    parser.add_argument('--train_url', type=str, default='./output')
    parser.add_argument('--base-dir', type=str, default='./code_test')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pretrained', type=bool, default=False)
    
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--print-interval', type=int, default=10)

    parser.add_argument('--temporal-size', type=int, default=100)
    parser.add_argument('--init_method'  )
    
    args = parser.parse_args()
    
    os.makedirs(model_dir)
    os.makedirs(data_dir)
    moxing.file.copy_parallel(args.data_url, data_dir) 
    
    
    main(args)
