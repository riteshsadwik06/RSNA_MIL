from __future__ import print_function
import wandb
import numpy as np
import pickle
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
# import dataloader 
from model import Attention, GatedAttention
from torch.utils.data import TensorDataset, DataLoader
import torch
import pdb
import dataloader
import pandas as pd
import glob




#########

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=56, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    #model.double()
    model.cuda()


wandb.init(project='MIL_trails',entity ='rsadwik')
config = wandb.config
config.learning_rate = args.lr
config.weight_decay = args.reg
config.epochs = args.epochs
#wandb.run.name = str(args.reg)+'_wd_'+str(args.epochs)+'_ep_'+str(args.lr)+'_lr'
#wandb.run.save()


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
# optimizer=optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=args.reg)

train_inputs_list, train_labels_list, test_inputs_list,test_labels_list = dataloader.train_test_load(args.seed)

train_loader = data_utils.DataLoader(dataloader.rsna(list_of_file_paths=train_inputs_list,labels_list=train_labels_list),
                                     batch_size=1,
                                     shuffle=True)
test_loader = data_utils.DataLoader(dataloader.rsna(list_of_file_paths=test_inputs_list,labels_list=test_labels_list),
                                     batch_size=1,
                                     shuffle=True)



wandb.watch(model)

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.squeeze(0)#train_inputs_list[i]
        bag_label = label#train_labels_list[i]

        # if(batch_idx==0):
        #     #print('printing data shape',data.shape)
        #     print(data.shape[1])

    # for i in range(len(train_inputs_list)):


        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

    # reset gradients

        
# calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        # if (i == 0):
            # print('no of slides',data.shape[1])
        loss = loss/(data.shape[1])

        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
    
        train_error += error
# backward pass
        loss.backward()
# step
        if(batch_idx%5==0):
            optimizer.step()
            optimizer.zero_grad()
            # print(loss)

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return train_loss,train_error


def test(epoch):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        # for batch_idx, (data, label) in enumerate(test_loader[i]):
        data = data.squeeze(0)
        bag_label = label
            #instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        loss, attention_weights = model.calculate_objective(data, bag_label)
        loss = loss/(data.shape[1])
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

  
    test_error /= len(test_loader)
    test_loss /= len(test_loader)


    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    return test_loss,test_error


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        print('epoch:',epoch)
        train(epoch)
        test(epoch)
        train_loss,train_error = train(epoch)
        test_loss,test_error = test(epoch)
        metrics = {'train_loss':train_loss,'train_error':train_error,'test_loss':test_loss,'test_error':test_error}
        wandb.log(metrics)
