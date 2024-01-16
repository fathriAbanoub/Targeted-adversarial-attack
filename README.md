# Targeted-adversarial-attack
Adversarial Traffic Light Image Generation

This code demonstrates the generation of adversarial traffic light images using a pre-trained ResNet model. The objective is to create an image of a green traffic light that is misclassified by the model, which originally recognizes it as a red traffic light.
Objective Function

The overall objective function to minimize during the optimization process is given by:

J(θ,x,y)=−CrossEntropyLoss(model(x),target_label)+Perceptual_LossJ(θ,x,y)=−CrossEntropyLoss(model(x),target_label)+Perceptual_Loss

This objective function comprises two terms:

    The negative of the cross-entropy loss encourages the model to predict the target class (yy).
    The perceptual loss ensures that the perturbed image remains visually similar to the original image.

Optimization Step

The optimization step involves updating the input image (xx) by adding a perturbation (δδ) calculated based on the gradient of the objective function. The perturbation is defined as:

δ=ϵ⋅sign(∇xJ(θ,x,y))δ=ϵ⋅sign(∇x​J(θ,x,y))

Here, ϵϵ is a small constant, and ∇xJ(θ,x,y)∇x​J(θ,x,y) is the gradient of the objective function with respect to the input image. The goal is to find an adversarial perturbation that misleads the model while minimizing perceptual differences.
Prerequisites

    PyTorch: Ensure that PyTorch is installed. You can install it using:

    bash

pip install torch torchvision

Matplotlib: Install Matplotlib for displaying images:

bash

    pip install matplotlib

Usage

    Load the pre-trained ResNet model and the original red traffic light image.
    Transform the original image to the desired format for the model.
    Define the target label corresponding to a green traffic light.
    Set up the loss function (cross-entropy) and the optimization algorithm (SGD).
    Iterate through a specified number of steps, generating adversarial examples using the Fast Gradient Sign Method (FGSM).
    Display the results, showing the original and perturbed images side by side.
    Check model predictions, confidence scores, and visualize the noise vector introduced during perturbation.

Parameters

    epsilon: A hyperparameter controlling the magnitude of the perturbation.
    num_steps: The number of optimization steps to generate the adversarial example.
    perceptual_loss_constraint: A constraint on the perceptual loss to ensure subtle visual changes.

Results

The generated adversarial image aims to visually resemble a green traffic light while being misclassified by the model. The code outputs confidence scores and noise vectors, providing insights into the adversarial perturbation.

Feel free to experiment with different hyperparameters and explore the impact on the adversarial image generation process.
