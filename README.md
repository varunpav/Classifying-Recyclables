# Classifying-Recyclables
Recyclable vs. Non-Recyclable Image analysis

Group members: Varun Pavuloori, Erin Moulton, Hank Dickerson

Check out the Project motivations in the [Hook Document](Recycling_HookDoc.pdf)

For any guidance check out the [Rubric](Rubric.pdf)

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

Clone in the data using the github website at the top of this README file.

Validate the structure using the following commands (!ls realwaste, !ls realwaste/RealWaste).

Set up the root directory of basepath where the waste data is stored.

Set up subdirectories of Recyclable and Non-Recyclable for organizing folders into recylable and non recylcable images.

Use the os.makedirs method to ensure the recyclable and non-recyclable folders were created.

Create a list recyclable_folders specifying which subfolders of waste items are recyclable.

Create a list non_recyclable_folders specifying which subfolders of waste items are non-recyclable.

Create a for loop for each folder in recyclable_folders.

Use src to construct the source path.

Use dst to construct the destination path.

Use the shutil.move method to transfer each subfolder from the base path to the recyclable folder previously made.

Repeat the previous four steps but this time for non_recyclable_folders.

Validate this new structure.

Define a function validate_images with a single input of folder_path

Define the valid image extensions of .jpg, .jpeg, and .png.

Use os.walk method to search through all subdirectories and files in the specified folder (folder_path.

If the image extension is not valid remove the image from the data.

If the image has a valid extension use the Image.open method to open the image.

Use img.verify to ensure the image isn't corrupted.

Call the validate_images function on the recyclable and non_recyclable folders.

Initialize image_paths equal to an empty list.

Initialize labels equal to an empty list.

Initialize waste_types equal to an empty list.

Define valid_extensions equal to the valid image extensions.

Create a for loop that iterates through the recyclable and non_recyclable folders, and use enumerate to assign 0 to recyclable and 1 to non_recyclable.

Define the folder_path using os.path.join.

Create a for loop within this for loop that searches through all files and subdirectories within the folders using th os.walk method.

Create a for loop within this for loop for each file in files.

If the image file ends in a valid extension generate the absolute path of the image and append it to the image_paths list.

using labels.append add the recyclability label of 0 or 1 depending on the parent folder.

Extract the waste type from the subdirectory name and append it to the waste_types list.

End all the for loops.

Create a dataframe using the three lists we just made with columns of image, labels, and wastetype.

From PIL import Image.

Initialize image_sizes equal to an empty directory.

Create a for loop going through each image path in the image column of the dataset that was made.

With Image.open open the image.

Set size equal to img.size.

If the image size already exists increment its count, if not add a new entry with a count of 1.

Stop this for loop.

Create a for loop that prints each unique image size along with the number of images of that same size.

If the length of image_sizes is equal to 1, print "all images have the same size", if not print "Images have varying sizes.

Using os.path.splitext(x)[1] split the filename into its filename and extension.

Use .lower() to ensure all image extensions are all consistent.

Add a new collumn format to the dataframe for storing the image extensions.

Count the number of times each unique file extension is used in the format column.

Store this result in format_distribution.

Import hashlib.

Define a function calculate_image_hash with a single input of image_path.

Compute the md5 hash of the of the files contents using hashlib.md5 then convert it to a hexidecimal string using .hexdigest.

Apply this function to the image column of the dataset.

Check for duplicates in the image_hash column.

If duplicates are found print the rows corresponding to the duplicates.

From PIL import Image.

Define a function check_image_mode with an input of image_path.

With Image.open and image_path open the image.

Return the image mode.

Create a new column image_mode in the dataset storing the colors of the images.

Count the number of images for each unique mode in the image_mode column.

Create a frequency table for the image column.

Split the data into traning and temporary sets.

Set the test size to .3 and the random state to 42

Split the temporary set into validation and test sets with a test size of .5.

From torchvision import transforms.

Preprocess the training images.

Resize all the images to 128X128.

Horazontally flip the images with a 50% propability.

Rotate the images randomly with a 10 degree range.

Using ToTensor convert the images to PyTorch tensor and scale pixel values to 0 or 1.

Scale pixel values using the datasets mean and standard deviation.

Preprocess the test images.

Resize all the images to 128X128.

Using ToTensor convert the images to PyTorch tensor and scale pixel values to 0 or 1.

Scale pixel values using the datasets mean and standard deviation.

From torch.utils.data import Dataset.

Define the WasteDataset class Dataset.

Initialize the dataset with self.dataframe and self.transform.

Define the dataset length.

Retrieve a single sample.

Using img_path and self.dataframe.iloc extracth the file path of the image.

Using label and self.dataframe.iloc retrieve the label corresponding to the image.

Using Image.open open the image and convert it to RGB.

Apply any specified transformations to the images like resizing or data augmentation.

Return image and its label.

From torch.utils.data import DataLoader.

Create the dataset instances for the train_dataset, val_dataset, and test_dataset using WasteDataset and transorm = transform_train.

Set the batch size equal 32.

Create the data loader instances using DataLoader for the train_loader, val_loader, and test_loader.

Define a function imshow with the input of img.

Unnormalize the images using torch.tensor and np.transpose.

Set data_iter equal to an iterator for the train_loader DataLoader.

Set images, labels equal to something that will get the next batch of images and labels from the iterator.

Using the imshow function display the first image of the batch.

Print the label corresponding to the first image.

From torchvision.models import resnet18.

Import torch.nn as nn.

Load the pretrained resnet18 model.

Set num_features equal to model.fc.in_features.

Set model.fc equal to nn.Linear with a binary classification.

Import torch.optim and optim.

Set criterion equal to nn.CrossEntropyLoss().

using the optim.Adam method create an optimizer.

Dynamically select a device.

If a GPU with CUDDA support is available use that, if not use a CPU.

Set model equal to model.to(device).

Using the torch.optim.Adam initialize an optimizer.

Set num_epochs equal to 20.

Create a for loop for epoch in range of num_epochs.

Set the model to training mode.

Set running_loss equal to 0.0.

Move the images and labels to the selected device.

Zero the gradients before the backward pass.

For the forward pass pass the images through the model.

Compute the loss.

For the backward pass calculate the loss backwards to compute the gradients.

Update the model weights based on the gradients.

Accumulate the loss for this batch.

Use the model.eval method.

Set val_loss equal to 0.0.

set correct equal to 0 and total equal to 0.

With torch.no_grad create a for loop that iterates over batches of images and labels from the validation set.

Move the validation labels and images to the selected device.

For the forward pass have the model compute the predicted output for the input images.

Use the loss function to compare the predicted output with the true labels.

Have the loss of the batch added to the total validation loss.

Use the torch.max method to compute the predicted class for each image.

Chech whether the predicted label matches the true label, if it does add 1 to correct.

After each image has been processed add 1 to the total.

Write a line that calculates the validation accuracy.

Print the results.

Repeat the previous 13 steps for the test set.

Use the torch.save method to retrieve the state dictionary of the model.

Create our own model.

Import tensorflow as tf.

Set the data directory to our image data.

Set the image height and width to 128.

Set the batch size to 32.

Use the image_dataset_from_directory method create a training dataset and load and labels images from the directory structure.

Set the validation split to .2.

Set the seed equal to 42.

Repeat the previous 3 steps for a validation dataset.

Print the class names.

From tensorflow.keras.models import Sequential

From tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout

From tensorflow.keras.regularizers import l2, l1_l2

Import keras

Set myModel equal to keras.models.sequential().

Add the first convulutional layer with 64 filters of size 5,5.

Add aditional convulutional layers that increases the number of filters to 128 then 256.

Add the flatten method to the model.

Add a fully connected layer with 256 units, using ReLU activation.

Use BatchNormalization to stabilize training.

Have another fully connected dense layer with 128 units with the other settings the same.

Add the output layer with 18 units (one for each waste category).

Use the summary method to summarize the model.

Set myEpochs equal to 20.

Using the keras.optimizer.SGD method initialize an optimizer with a learning rate of 0.0001.

Set myLoss equal to "sparse_categorical_crossentropy".

Set myMetrics equal to a list including 'accuracy'.

Use the myModel.compile method to configure the model for training.

Use the myModel.fit method to train the model using the training data.

# References
[1] sam-single. (2023). GitHub - sam-single/realwaste: RealWaste is an image dataset of waste items in their authentic state. GitHub. https://github.com/sam-single/realwaste?tab=readme-ov-file

[2] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). "ImageNet: A Large-Scale Hierarchical Image Database." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
