import argparse
import os
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import random
from resnet20 import *
from alexnet import AlexNet
from tqdm import tqdm
from torchvision.transforms import transforms as T

def eval(args,test_loader,model):
    # model=torch.load(args.ckpt+'model.pt',map_location='cuda')
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().cpu().item()
    print('Accuracy : %d %%' % (100 * correct / total))

# def train(args,epoch,train_loader,model,optimizer,lr_scheduler,criterion):
#     for e in range(epoch):
#         model.train()
#         losses=[]
#         for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#             data, target = data.cuda(), target.cuda()
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             # print('Step : {}, Loss : {:.4f}'.format(batch_idx,loss.cpu().item()))
#             losses.append(loss.cpu().item())
        
#         lr_scheduler.step(e)
#         print('Epoch : {}, Loss Avg : {:.4f}'.format(e+1,losses.mean()))
#         torch.save(model,args.ckpt+'model.pt')
def train2(args,e,train_loader,model,optimizer,criterion):
    model.train()
    losses=[]
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # print('Step : {}, Loss : {:.4f}'.format(batch_idx,loss.cpu().item()))
        losses.append(loss.cpu().item())
    
    # lr_scheduler.step(e)
    print('Epoch : {}, Loss Avg : {:.4f}'.format(e+1,sum(losses)/len(losses)))
    # torch.save(model,args.ckpt_dir+'model.pt')

def set_seed(seed=42):
    random_seed=seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main(args):
    ## Check Augmentation settings
    train_transform = T.Compose([T.RandomHorizontalFlip(),
                      T.Pad(padding=4, padding_mode='reflect'),
                      T.RandomCrop(32, padding=0),
                      T.ToTensor(),
                      T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
     

    test_transform = T.Compose([
                           T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                          )
    # transform=T.Compose([T.Resize(224),
    #                             T.ToTensor(),
    #                             T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)

    test_dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True)
    
    print('Quantization Bits\n')
    print('Weight: {}\nActivation: {}\nGradient: {}\nGradient_Quantize : {}'.format(args.W, args.A, args.G,args.gradient_quantize))
    # Reducing Runtime Memory
    rrm=False
    if args.RRM:
        rrm=True
    # model=AlexNet(args.W, args.A, args.G,10,rrm,args.gradient_quantize).cuda()
    model=resnet20(args.W, args.A, args.G,10,rrm,args.gradient_quantize).cuda()
    
    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
    lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1)
    criterion=torch.nn.CrossEntropyLoss().cuda()
    epoch=args.epoch
    for e in range(epoch):
        lr_scheduler.step(e)
        train2(args,e,train_loader,model,optimizer,criterion)
        # train(args,epoch,train_loader,model,optimizer,lr_scheduler,criterion)
        eval(args,test_loader,model)

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')
    parser.add_argument('--epoch', default=100, type=int, help='Total Epochs')
    parser.add_argument('--W', default=1, type=int, help='Weight Quantization bit')
    parser.add_argument('--A', default=2, type=int, help='Activation Quantization bit')
    parser.add_argument('--G', default=6, type=int, help='Gradient Quantization bit')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--RRM', action='store_true', help='Reducing Runtime Memory : Paper Section 2.8')
    parser.add_argument('--gradient_quantize', default=False, type=bool, help='Gradient Quantization')


    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    # parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    parser.add_argument('--log_dir', default='./log', type=str, help='log_directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='checkpoint_directory')
    
    args=parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    set_seed(42)
    main(args)
