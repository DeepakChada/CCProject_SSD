bucket_name = "imageforgery"
project_name = "My First Project"
def get_gcsfs(project_name):
import gcsfs
import google
import json
google.colab.auth.authenticate_user()
creds, project = google.auth.default()
gcs = gcsfs.GCSFileSystem(token=creds, project=project_name)
return json.loads(gcs.credentials.to_json())
!pip install -q bytehub[aws]
import pandas as pd
import numpy as np
import os
import shutil
import bytehub as bh
print(f'ByteHub version {bh.__version__}')
fs = bh.FeatureStore()
fs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from joblib import dump, load
plt.ion() # interactive mode
"""# Load Dataset
x = dataitems
y = 1 | Tampered
y = 0 | Not Tampered
"""
transform = transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model_conv = torchvision.models.alexnet(pretrained=True)
classifier = list(model_conv.classifier.children())
model_conv.classifier = nn.Sequential(*classifier[:-1])
for param in model_conv.parameters():
param.requires_grad = False
from google.colab import drive
drive.mount('/content/drive')
x = []
y = []
model_conv.eval()
for i in range(1,2):
scales = None
for scale_img in os.listdir(f'drive/MyDrive/MICC-F220-labeled/{i}scale'):
img = Image.open(f'drive/MyDrive/MICC-F220-
labeled/{i}scale/{scale_img}')
img_tensor = transform(img)
img_tensor.unsqueeze_(0)
scale_ftrs = model_conv(img_tensor)
scale_ftrs.squeeze_(0)
scales = scale_ftrs.cpu().numpy()
x.append(np.concatenate((scales, scales)))
y.append(0)
for tamp_img in os.listdir(f'drive/MyDrive/MICC-F220-labeled/{i}tamp'):
img = Image.open(f'drive/MyDrive/MICC-F220-
labeled/{i}tamp/{tamp_img}')
img_tensor = transform(img)
img_tensor.unsqueeze_(0)
tamp_ftrs = model_conv(img_tensor)
tamp_ftrs.squeeze_(0)
tamp_ftrs = tamp_ftrs.cpu().numpy()
x.append(np.concatenate((scales, tamp_ftrs)))
y.append(1)
x = np.array(x)
y = np.array(y)
x.shape, y.shape
np.unique(y, return_counts=True)
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(x,y)
import sklearn
sklearn. __version__
classifier.score(x,y)*100
classifier.predict(x)
torch.save(model_conv, 'alex.pkl')
dump(classifier, 'svm.joblib')
"""# Inference"""
def predict(img1_path, img2_path, ftr_ext_path, classifier_path):
transform = transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
alexnet = torch.load(ftr_ext_path)
alexnet.eval()
classifier = load(classifier_path)
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)
img1_tensor = transform(img1)
img1_tensor.unsqueeze_(0)
img1_ftrs = alexnet(img1_tensor)
img1_ftrs.squeeze_(0)
img1_ftrs = img1_ftrs.cpu().numpy()
img2_tensor = transform(img2)
img2_tensor.unsqueeze_(0)
img2_ftrs = alexnet(img2_tensor)
img2_ftrs.squeeze_(0)
img2_ftrs = img2_ftrs.cpu().numpy()
x = np.concatenate((img1_ftrs, img2_ftrs))
x = np.expand_dims(x, axis=0)
return classifier.predict(x)
p=predict('drive/MyDrive/MICC-F220-
labeled/1scale/CRW_4853_scale.jpg',
'drive/MyDrive/MICC-F220-labeled/1scale/CRW_4853_scale.jpg',
'alex.pkl', 'svm.joblib')
#if p==0:
print(p)
p1=predict('drive/MyDrive/MICC-F220-
labeled/1scale/CRW_4853_scale.jpg',
'drive/MyDrive/MICC-F220-labeled/1tamp/CRW_4853tamp132.jpg',
'alex.pkl', 'svm.joblib')
print(p1)
