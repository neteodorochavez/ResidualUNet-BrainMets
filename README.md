# BrainMet Image Segmentation w/ Residual 3D UNet

In this repo you will find notebooks and scripts which will allow you to find all the code necessary to replace the experiments conducted in this project. This Residual UNet 3D Model is trained on 693 patients and aims to be as simple as possible in terms of code readability and project structure. Improvements, suggestions, and feedback are welcome. 

Main libary of the project is a customer Residual UNet 3D Model. Initially original code experiments were written in an older version of PyTorch. Workflow adapted in this repo is pretty generic and potentially can be used for other similar projects with different data or task with minimal changes. 

This project leverages PyTorch, PyTorch Lightning, and Neptune.ai to train, evaluate, and monitor progress. The model has been trained to optimize the [Dice Score Coefficient] (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) . It has been hyperparameter tuned and a final diagnosis of its performance across various sized brain mets is displayed. 

The ambition of this project was to increase the model performance for small brain metastasis. 

Currently, there are additional improvements to model training that is still in development. 
- Augmentation (Mix N Match Approach with 3D Volumes) 
- Hyperparameter Tuning 
- Random Noise
This repo is still in development. 
