#Things to watch out for- argparse, gpu vs cpu, importing properly


#import statements
#import formulas.py

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
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
import json

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('path_to_image', type=str, help='Store a path to image')
parser.add_argument('checkpoint', type=str, help='Store the path to the saved model')
parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int, help='Store a value for the number of most likely classes for the image prediction')
parser.add_argument('--category_names', action='store', type=str, dest='category_name', default="/home/workspace/aipnd-project/cat_to_name.json", help='Store the path to the saved model')
parser.add_argument('--gpu', action='store_true', dest='is_gpu', default=False, help='Store gpu') #

results = parser.parse_args()
path_to_image = results.path_to_image
checkpoint = results.checkpoint
top_k = results.top_k
category_name = results.category_name
is_gpu = results.is_gpu

print("Path to image: " + path_to_image)
print("Checkpoint: " + checkpoint)
print("Number of most likely classes in output: " + str(top_k))
print("Path to category-mapping json file: " + category_name)
print("Using gpu: " + str(is_gpu))




#Sanity checks
if not os.path.isfile(path_to_image):
    sys.exit("Cannot find image '{}'".format(path_to_image))

if not os.path.isfile(checkpoint):
    sys.exit("Cannot find model '{}'".format(checkpoint))

if not os.path.isfile(category_name):
    sys.exit("Cannot find file '{}'".format(category_name))
    
#Load model
print("Loading model '{}'".format(checkpoint))

try:
    state_dict = torch.load(checkpoint)
except:
    sys.exit("Cannot load model '{}'".format(checkpoint))
class_to_idx = state_dict['class_to_idx']
learning_rate = state_dict['learning_rate']

model = formulas.build_model(state_dict['arch'], state_dict['hidden_units'], class_to_idx)
model.load_state_dict(state_dict['state_dict'])
optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
optimizer.load_state_dict(state_dict['optimizer'])

print("Loaded '{}' (arch = {}, epochs = {})".format(checkpoint, state_dict['arch'], state_dict['epochs']))




#Predict image (process image)
probabilities, classes = formulas.predict(path_to_image, model, top_k, is_gpu)



#Print results
    #print(probabilities)
    #print(classes)




#Extra?
try:
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
except:
    sys.exit("Cannot load category-mapping json file '{}'".format(category_name))
    
for index in range(len(probabilities)):
    print("{}: {}".format(cat_to_name[classes[index]], probabilities[index]))
