**Objective :** This project aims to develop a deep learning model to identify whether an image contains a forest fire using a convolutional neural network (CNN). The model is trained using a dataset of forest fire images and evaluated for its accuracy and loss.

**Data Preprocessing and Augmentation :**

The project begins by importing necessary libraries, including TensorFlow and Keras for model creation and data manipulation. The dataset for training and testing the model is stored in Google Drive.

The ImageDataGenerator class from Keras is used to perform data augmentation on the training set. This includes:

Rescaling the pixel values to the range [0, 1].

Shear transformations and zoom to increase variability.

Horizontal flipping for further augmentation.

**Model Architecture :** 

The model uses a VGG16 pre-trained model, which is loaded without its top classification layer. The base of VGG16 is frozen to retain the learned features, and new fully connected layers are added for the classification task. The network structure is as follows:

Base Model: VGG16 (with include_top=False) to extract feature representations from images.

Additional Layers: MaxPooling2D layer for downsampling.

Flatten to transform the 2D feature maps into 1D vectors.

Dense layers with increasing complexity: 1520 units, 750 units, 64 units, 32 units.

A Dropout layer (rate=0.5) to prevent overfitting.

The output layer consists of a single neuron with a sigmoid activation function for binary classification.

**Model Compilation and Training :**

The model is compiled with the binary crossentropy loss function and Adam optimizer with a learning rate of 0.0001. The model is trained for 10 epochs using the training generator for batches of 16 images at a time. The training process is monitored by evaluating both training accuracy and validation accuracy.

**Evaluation and Results :**

The model's performance is evaluated on the validation set, yielding the following results:

Test Loss: 0.2519

Test Accuracy: 93.95%

This indicates that the model is performing well with a high level of accuracy.

**Visualization :**

To understand the model's learning over time, two plots are generated to visualize:

Training vs Validation Accuracy – Displays how accuracy improves for both training and validation sets.

Training vs Validation Loss – Shows the loss trend for both sets, which is essential for detecting overfitting or underfitting.

**Model Deployment using Gradio:**

To make the model more accessible, a Gradio interface is created, allowing users to upload an image and receive a prediction (Fire or Non-Fire). The image is preprocessed (resized, normalized) before being passed to the model for classification. The output is a label indicating whether a forest fire is detected.

**Conclusion :**

The model successfully detects forest fires with an accuracy of 93.95% after 10 epochs. By leveraging transfer learning with the VGG16 pre-trained model and applying appropriate data augmentation and regularization techniques, the model is able to generalize well. The Gradio interface makes it easy for users to interact with the model and make predictions on new images.
