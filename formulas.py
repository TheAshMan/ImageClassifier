#Import statements
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
from PIL import Image


#Define classifier
def build_model(arch, hidden_units, class_to_idx): 
    #TODO add architecture as an input, and use it to define the architecture in the following lines
    
    #Define model architecture
    input_layers = 10000
    
    if arch == "alexnet":
        model = models.alexnet(pretrained = True)
        input_layers = 9216
        print("Building 'alexnet' model")
    else:
        model = models.vgg16(pretrained=True)
        input_layers = 25088
        print("Building 'vgg' model")
    
    #Freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    #Define model layers 
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layers, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    #Setting the classifier
    model.classifier = classifier
    
    #Setting the training data
    model.class_to_idx = class_to_idx
    
    #Return model
    return model



#Train model
def train(model, epochs, criterion, optimizer, trainloader, validloader, is_gpu): #TODO change gpu to cpu where needed
    
    #Every 40 steps, display the current model accuracy, loss, etc
    print_every = 40
    steps = 0

    # change to cuda for gpu if available and selected
    cuda = torch.cuda.is_available()     
    if is_gpu and cuda:
        model.to('cuda')

    #Begin looping
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            if is_gpu and cuda:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            optimizer.zero_grad() #Might have something to do with resnet18

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #Display the status of the model
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate(model, criterion, validloader, is_gpu)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(valid_accuracy))

                running_loss = 0

#Validate method
def validate(model, criterion, data_loader, is_gpu):
    model.eval()
    accuracy = 0
    loss = 0
    
    for inputs, labels in iter(data_loader):
        #loss
        cuda = torch.cuda.is_available()     
        if is_gpu and cuda:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            
        output = model.forward(inputs)
        loss += criterion(output, labels).item()
        
        #accuracy
        expected_data = torch.exp(output).data 
        equality = (labels.data == expected_data.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    return loss/len(data_loader), accuracy/len(data_loader)    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    new_size = [0, 0]

    if image.size[0] > image.size[1]:
        new_size = [image.size[0], 256]
    else:
        new_size = [256, image.size[1]]
    
    image.thumbnail(new_size, Image.ANTIALIAS)
    width, height = image.size  

    left = 256/2 - 224/2
    right = 256/2 + 224/2
    top = 256/2 - 224/2
    bottom = 256/2 + 224/2
    
    image = image.crop((left,top, right, bottom))    

    
    np_image = np.array(image)
    np_image = np_image/255.
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image_transposed = np_image.transpose(2,0,1)
    
    return np_image_transposed



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk, is_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    try:
        np_array = process_image(Image.open(image_path))
    except:
        sys.exit("Cannot open image file '{}'".format(image_path))
    tensor = torch.from_numpy(np_array)
    
    cuda = torch.cuda.is_available()     
    if cuda and is_gpu: #if gpu is available and it is chosen
        inputs = Variable(tensor.float().cuda())
    else:       
        inputs = Variable(tensor)
    inputs = inputs.unsqueeze(0)
        
    
    with torch.no_grad():
        output = model.forward(inputs.cpu()) #not sure if this should have .cpu()
    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
        
    return probabilities.numpy()[0], mapped_classes	