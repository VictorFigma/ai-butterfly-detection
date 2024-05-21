# 🦋 Butterfly Detection Challenge

## 📸 Context

This project was created as part of a job application challenge to showcase my skills. It utilizes a dataset that cannot be shared due to (potential) confidentiality reasons, and no PRs are allowed.

## 📁 How to execute 

 1. Insert a butterflies dataset into the "/data/test/" directory for testing data and the "/data/train/" directory for training data.

2. Install
    ```
    bash install.sh
    ```

3. Run: train, validate and predict.
    ```
    python model.py
    ```

## 📊 Results


The trained model currently achieves ~99.5% accuracy. With further parameter tweaking improvement is feasible since extensive optimization hasn't been performed.
# About the challenge

## 🎯 Objectives

"Your task is to create a neural network model that can process images from forest cameras and accurately detect the presence of butterflies. This model must differentiate butterflies from other insects, adapting to various lighting conditions and angles."

## 📁 Dataset

"You will be provided with a dataset comprising images taken from the forest, with various scenes including different animals, plants, and insects. Some images will contain butterflies, while others will not.

The `predictions` folder will contain the `predictions.json` file with your model's predictions on whether an image contains a butterfly or not."

## 🎯 Tasks

"Develop a neural network to process and detect butterflies in images from forest cameras, contributing to the biologists' research efforts."

## 📊 Data Processing

"Data preprocessing should be applied to normalize and prepare the images for the model, considering the various lighting conditions and angles present in the dataset."