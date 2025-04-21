# Generating a Neural Network for CIFAR-10 Classification Using VAE-Diffusion
<!-- 
## Overview
This project uses a combination of Variational Autoencoders (VAE) and Diffusion Models to generate fully trained neural networks for CIFAR-10 image classification. Instead of training on data, we're training on model weights themselves.

## Features
- VAE for encoding neural network weights into a latent space
- Diffusion model for generating novel neural network weights
- Training pipeline for both VAE and diffusion models
- Evaluation tools for generated models
- Support for both direct weight diffusion and latent diffusion approaches

## Requirements
- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm

## Installation
```bash
git clone https://github.com/your-username/diffusion-data-aug.git
cd diffusion-data-aug
pip install -r requirements.txt  # If you have a requirements file
```

## Usage

### Training the VAE
```bash
python train_vae.py --checkpoint_dir Toy_CNN --batch_size 32 --epochs 100
```

### Training the Diffusion Model
```bash
python train_diffusion.py --checkpoint_dir Toy_CNN --batch_size 8 --epochs 1000
```

### Generating New Models
```bash
python sample.py --model_path path/to/diffusion_model.pth --vae_path path/to/vae_model.pth
```

### Evaluating Generated Models
```bash
python evaluate_generated.py --model_path path/to/generated_model.pth
```

## Project Structure
- `vae_model.py`: Implementation of the Variational Autoencoder for model weights
- `diffusion_model.py`: Implementation of the Diffusion Model
- `vae_diffusion.py`: Implementation of Latent Diffusion that works in the VAE latent space
- `train_vae.py`: Training script for the VAE
- `train_diffusion.py`: Training script for the Diffusion Model
- `diffusion_trainer.py`: Training utilities for diffusion models
- `sample.py`: Generate new models using the trained diffusion model
- `evaluate_*.py`: Evaluation scripts for generated models
- `Gen_Diffusion_Dataset.py`: Tools for creating training datasets


## Authors
- @Swahaha
- @rngtang -->

## Project Overview
* Our project aims to use a combination of a VAE model and a Denoising Diffusion Probabilistic Model to generate fully trained neural networks for classifying the CIFAR-10 dataset. Instead of training the NN directly, we are generating the weights using the Diffusion model.

## Background
* Traditionally, Diffusion models are used to generate images. But, diffusion models can be used for wide variety of applications like audio, nlp, or even Neural Networks.

## Methodology
* Our project is divided into three main components: generating dataset, training model, evaluating results.

### Generating dataset
* To be able to generate the training dataset, we have a simple NN with 10 layers.
* TinyNN is trained with the CIFAR-10 dataset a total of 10 times with a max of 500 epochs each time. 
* For each instance, it generates 250 checkpoints with accuracy > 0.75
* Each time it trains the TinyNN, it does a random data augmentation (random horizontal flip, random crop padding, ColorJitter)
* At the end, there is a total of 2500 checkpoints.
* For the base model, it is run 4 times in parallel to create a total of 10,000 checkpoints.

### Training model
The model training is divided in two parts: training VAE and training Diffusion.
#### Training VAE
The main purpose of this part is to train a VAE model that learns a latent representation of CNN model weights. The training works the following way:
* Creates dummy TinyNN to get total number of parameters
* Uses the WeightDataset to load flattened weights from all checkpoints
* The model takes flattened weights as input and learns to compress them into a latent space and then reconstruct them.
* Uses Adam and StepLR scheduler for learning rate decay.
* During training, tracks the best performance and saves it as a checkpoint. 
* Every 10 epochs, samples a new weight vector from the VAE and reconstructs a new TinyNN
-->

## Training Diffusion
This model is trained to generate neural network weights using a diffusion model that operates in the latent space of the VAE trained. The model training the following way:
* Loads the TinnyCNN checkpoints, flattens and encodes using the VAE encoder previously trained. Produces of tensor of latent vectors
* Trains the diffusion model on the VAE latent vectors. 
* For each vector, adds random noise and trains the model to predict that noise using MSE loss.
* Generates new model by sampling a random noise in the latent space, applies reverse diffusion, gets latent vector, decodes via VAE to get the new model weights.

## Evaluating Results
After training the VAE and Diffusion model, we evaluate the performance in generating NN capable of classifiying CIFAR-10 images. The evaluation has the following workflow
* Sample a latent vector from the diffusion model
* Decode it into a weight vector using the VAE and reconstructs a TinyNN 
* Evaluates the generated TinyCNN on the CIFAR-10

## Experiments
Here is a list of possible things we could explore
* How do different datasets affect it accuracy (sparse, regularization, etc)?
* Do the number of samples have an impact? Are the generated NN in teh dataset or are they newly generated?
* How do the training epochs affect the performance?

## Conclusions
* It works!!!!
