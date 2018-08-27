

#import statements


import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import formulas
import argparse
import os
import sys

#argparse dataset, saving checkpoint in directory, architecture, hyperparameters(epochs,learning rate, hidden units (so 1 hidden layer)), gpu
parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help='Store a directory')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16', help='Store architecture: \'vgg16\', \'alexnet\'')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='/home/workspace/paind-project', help='Store save directory')
parser.add_argument('--learning_rate', action='store', type=float, dest='learning_rate', default=0.001, help='Store learning rate')
parser.add_argument('--hidden_units', action='store', type=int, dest='hidden_units', default=1600, help='Store hidden units')
parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=5, help='Store epochs')
parser.add_argument('--gpu', action='store_true', dest='is_gpu', default=False, help='Store gpu') #



results = parser.parse_args()
data_dir = results.directory
arch = results.arch
save_directory = results.save_dir
learning_rate = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
is_gpu = results.is_gpu
print("Data directory: " + data_dir)
print("Model architecture: " + arch)
print("Save directory: " + save_directory)
print("Learning rate: " + str(learning_rate))
print("Number of hidden units: " + str(hidden_units))
print("Number of epochs: " + str(epochs))
print("Using gpu: " + str(is_gpu))

#Sanity checks?
if not os.path.isdir(data_dir): #if dataset directory does not exist, stop program
    sys.exit("Error: Invalid dataset directory \"" + data_dir + "\"")

if not os.path.isdir(save_directory): #if save directory does not exist, stop program
    sys.exit("Error: \"--save_dir\" is followed by an invalid save directory \"" + save_directory + "\"")

#Get directories
if data_dir[len(data_dir)-1] is not "/": #If directory has "/" at the end or not, this will format so it doesn't matter
    data_dir += "/"
    #print(data_dir)

train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'
test_dir = data_dir + 'test'

    #print(train_dir + " " + valid_dir + " " + test_dir)
    #Sanity checking subdirectories
if not os.path.isdir(train_dir): #if train directory does not exist, stop program
    sys.exit("Error: Dataset directory does not contain subdirectory \"" + train_dir + 
             "\":\nPlease make sure that the dataset directory is divided into 3 subdirectories of names 'train', 'valid', and 'test'")
if not os.path.isdir(valid_dir): #if valid directory does not exist, stop program
    sys.exit("Error: Dataset directory does not contain subdirectory \"" + valid_dir + 
             "\":\nPlease make sure that the dataset directory is divided into 3 subdirectories of names 'train', 'valid', and 'test'")
if not os.path.isdir(test_dir): #if test directory does not exist, stop program
    sys.exit("Error: Dataset directory does not contain subdirectory \"" + test_dir + 
             "\":\nPlease make sure that the dataset directory is divided into 3 subdirectories of names 'train', 'valid', and 'test'")    
    
    
#Transforms, datasets, dataloaders etc
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])    

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
test_data = datasets.ImageFolder(test_dir,transform = test_transforms)

    #print("Train data image count: " + str(len(train_data)))
    #print("Validate data image count: " + str(len(valid_data)))
    #print("Test data image count: " + str(len(test_data)))

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(train_data, batch_size=32)


#Build Model
class_to_idx = train_data.class_to_idx
model = formulas.build_model(arch, hidden_units, class_to_idx)
    #print(model)



#Train, validate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
print("Beginning training...")
formulas.train(model, epochs, criterion, optimizer, trainloader, validloader, is_gpu)
print("Training complete!")

#Save checkpoint
state = {
    'arch': arch,
    'learning_rate': learning_rate,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : train_data.class_to_idx,
    'hidden_units' : hidden_units
}

if save_directory[len(save_directory)-1] is not "/": #If save directory has "/" at the end or not, this will format so it doesn't matter
    save_directory += "/"
    #print(save_directory)

torch.save(state, save_directory + "checkpoint.pth")
print("Model saved!")
#