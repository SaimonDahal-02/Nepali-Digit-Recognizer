
# Nepali Digit MNIST using PyTorch

This project focuses on classifying Nepali digit images using a Convolutional Neural Network (CNN) implemented in PyTorch. It involves data preparation, model architecture design, training strategies, experiment logging, and performance evaluation.

## Data Format:

The dataset comprises images of Nepali digits. Each image is grayscale and is expected to be in a standard format suitable for image classification tasks. The dataset is divided into training and validation sets, with a split ratio of 80/20.

## Technologies Used:

- **Python**: Programming language used for implementation.
- **PyTorch**: Deep learning framework utilized for building and training the CNN model.
- **MLflow**: Experiment tracking platform used for logging and visualizing experiments.
- **tqdm**: Python library for adding progress bars to loops for better visualization during training.

## Model Architecture:

The CNN architecture is designed as follows:

### Convolutional Layers:

```
+---------------+---------------+--------------+--------------+--------------+
|   Layer Type  |   Input Size  |  Output Size | Kernel Size  |  Activation  |
+---------------+---------------+--------------+--------------+--------------+
|   Convolution |  1x28x28      | 32x26x26     | 3x3          |     GELU     |
+---------------+---------------+--------------+--------------+--------------+
|   Convolution |  32x26x26     | 32x24x24     | 3x3          |     GELU     |
+---------------+---------------+--------------+--------------+--------------+
|   Max Pooling |  32x24x24     | 32x12x12     | 2x2          |     None     |
+---------------+---------------+--------------+--------------+--------------+
```

### Linear Layers:

```
+-----------------+--------------+-------------+--------------+
|    Layer Type    |  Input Size  | Output Size |  Activation |
+-----------------+--------------+-------------+--------------+
|   Flatten        | 32x12x12     |   4608      |     None    |
+-----------------+--------------+-------------+--------------+
|   Fully Connected|   4608       |    512      |     GELU    |
+-----------------+--------------+-------------+--------------+
|   Fully Connected|   512        |     64      |     GELU    |
+-----------------+--------------+-------------+--------------+
|   Fully Connected|    64        |     10      |     None    |
+-----------------+--------------+-------------+--------------+
```

## Training Strategies:

- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Stochastic Gradient Descent (SGD), Adam, Adagrad
  - Initial Learning Rate: 0.001
  - Momentum: 0.9
- **Activation Function**: ReLU, GELU
- **Early Stopping**: Monitored validation loss and implemented early stopping mechanism with learning rate reduction strategy.

## Experiment Logging:

- **MLflow**: Experiment tracking and management tool used to log and visualize experiment metrics, including:
  - Validation loss
  - Training loss
  - Training accuracy
  - Validation accuracy


| Momentum | Min Delta | Learning Rate | Patience | Epochs | Weight Initialization | Activation Function | Optimizer Name | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|----------|-----------|---------------|----------|--------|-----------------------|---------------------|----------------|-------------------|---------------|---------------------|-----------------|
| N/A | 0.001 | 0.001 | 3 | 98 | xavier_normal | ReLU | Adagrad | 0.99397 | 0.023098 | 0.98970 | 0.03711 |
| N/A | 0.001 | 0.001 | 3 | 42 | xavier_normal | ReLU | Adam | 0.9997 | 0.00006 | 0.99588 | 0.01396 |
| N/A | 0.001 | 0.001 | 3 | 41 | xavier_normal | GELU | Adagrad | 0.99191 | 0.0255 | 0.99764 | 0.0483 |
| N/A | 0.001 | 0.001 | 3 | 26 | xavier_normal | GELU | Adam | 0.99985 | 0.00081 | 0.99382 | 0.02698 |
| 0.9 | 0.001 | 0.001 | 3 | 97 | xavier_normal | GELU | SGD | 0.99625 | 0.0130 | 0.99176 | 0.0326 |
| 0.9 | 0.001 | 0.001 | 3 | 100 | xavier_normal | GELU | SGD | 0.99522 | 0.0163 | 0.99235 | 0.0433 |
| 0.9 | 0.001 | 0.001 | 3 | 50 | xavier_normal | GELU | SGD | 0.99551 | 0.0154 | 0.990 | 0.0404 |
| 0.9 | 0.001 | 0.001 | 3 | 50 | xavier_normal | ReLU | SGD | 0.99257 | 0.0220 | 0.9888 | 0.0610 |
| 0.9 | 0.1   | 0.001 | 5 | 50 | N/A               | N/A | SGD | 0.99205 | 0.0229 | 0.9871 | 0.0490 |



## Performance Evaluation:

- The trained model achieved a test accuracy of 99.36% using Adam + GELU.
- The trained model achieved a test accuracy of 98.5% using Adagrad + GELU.
- The trained model achieved a test accuracy of 98.2% using SGD + GELU.
- The trained model achieved a test accuracy of 98.43% using Adagrad + ReLU.
- The trained model achieved a test accuracy of 97.9% using Adam + ReLU.

## Visualization:

**Using Adam + GELU**

![Plots](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adam_plot.png)

![Confusion Matrix](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adam_confusion_matrix.png)

**Using Adam + ReLU**

![Plots](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adam_relu_plot.png)

![Confusion Matrix](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adagrad_relu_confusion_matrix.png)

**Using SGD + GELU**

![Plots](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/plot2.png)

![Confusion Matrix](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/sgd_confusion_matrix.png)

**Using Adagrad + GELU**

![Plots](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adagrad_plot.png)

![Confusion Matrix](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adagrad_confusion_matrix.png)

**Using Adagrad + ReLU**

![Plots](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adagrad_relu_plot.png)

![Confusion Matrix](https://github.com/SaimonDahal-02/Nepali-Digit-Recognizer/blob/d6353bba73cca9cb1663a3aff4e4fb7eafaebeac/plots/adagrad_relu_confusion_matrix.png)


## Conclusion

Based on the experiments conducted, it can be concluded that the model using GELU activation function and Adam optimizer achieved superior performance. This model converged faster, reaching satisfactory validation accuracy of 0.99382 and a remarkable test accuracy of 99.366% in just 26 epochs. 

The key findings are summarized as follows:

- **Activation Function**: GELU activation function provided favorable results, contributing to the model's faster convergence.
- **Optimizer**: Adam optimizer demonstrated effectiveness in optimizing the model parameters, leading to improved performance.
- **Training Accuracy**: The model achieved an impressive training accuracy of 99.985%.
- **Validation Accuracy**: Validation accuracy reached a high of 99.382%, indicating good generalization of the model.
- **Test Accuracy**: The model achieved an outstanding test accuracy of 99.366%.

Considering the combination of fast convergence and high accuracy on both validation and test sets, the model utilizing GELU activation function and Adam optimizer is concluded to be the best-performing model for classifying Nepali digit images.