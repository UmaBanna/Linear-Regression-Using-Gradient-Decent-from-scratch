# Linear Regression with Gradient Descent

This project implements a simple linear regression model using gradient descent optimization. The program generates synthetic data, trains the model, and visualizes the regression line as it updates over each epoch.

## Prerequisites

Ensure you have Python 3 and the following libraries installed:
- Numpy
- Matplotlib

## Installation

You can install the required libraries using pip:

```bash
pip install numpy matplotlib

```
## Running the Program

To run the program, use the command:

``` python
python3 linear_regression.py

```

## File Structure

- linear_regression.py: Main script containing the implementation of the linear regression model.

## Linear Regression Model

The Linear_Regression class implements the linear regression model with the following methods:

- __init__(self, X, y, learning_rate=0.000001, epoches=500): Initializes the model with input data X, target data y, learning rate, and number of epochs.

- accumulated_error(self, y, y_pred): Calculates the accumulated errors for bias and weights.

- train_regressor(self): Trains the linear regressor using gradient descent optimization.

## Usage

The script generates synthetic data, initializes the linear regression model, and trains it. It also prints the mean square error (MSE) for each epoch and visualizes the regression line using an animation.

### Synthetic Data Generation

- x_values: Randomly generated input data.
- w: True weight.
- b: True bias.
- noise: Random noise added to the data.
- y_values: Generated target data with noise.

### Model Training and Visualization.

The model is trained using gradient descent, and the MSE for each epoch is printed. An animation is created to visualize the regression line as it updates over each epoch.

## Example Output

The script prints the MSE for each epoch and shows an animation of the training process, illustrating how the regression line converges to the best fit line over time.

```
plaintext

Mean Square Error for each epoch
Epoch 0: MSE=...
Epoch 1: MSE=...
...
```

## Author

This program is developed by Uma Maheshwari Banna