# Classifying-Recyclables
Recyclable vs. Non-Recyclable Image analysis

Group members: Varun Pavuloori, Erin Moulton, Hank Dickerson

Work completed as a part of the Data Science Project course @ UVA

Data sourced from: https://github.com/sam-single/realwaste
# Section 1: Software and Platform Section
Software used: Google Colab

Add-on Packages needed: os, shutil, PIL, sklearn.model_selection, torchvision, torch.utils.data, pandas, matplotlib.pyplot, torch, hashlib, seaborn, matplotlib.image, numpy, torchvision.models, torch.nn, torch.optim, tensorflow, tensorflow.keras.regularizers, tensorflow.keras.layers, tensorflow.keras.models, keras

Platforms: Windows, Mac
# Section 2: Project Folder Contents
Our project folder is composed of a Data folder an Output folder a Scripts folder a License and this README.md.

The Data folder contains a file showing where to find the dataset used for this project. There is also a data appendix showing statistics for all of the variables in the dataframe that we used to complete this project.

The Output folder contains various graphs and images. 

The Scripts folder includes a file of our source code used to obtain all of the results we observed throughout this project.

The License file is a simple copyright license.

# Section 3: Insturctions to Replicate Results
Set up a notebook using google colab.

Import os, shutil, pandas, matplotlib.pyplot, and torch.

From sklearn.model_selection import train_test_split.

From torchvision import transforms.

From torch.utils.data import DataLoader, Dataset.

