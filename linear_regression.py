""" Prerequisites: Python3, Numpy Library, Matplotlib Library """
""" To run the program please use the command " python3 linear_regression.py " from the current directory """

import numpy as np  # Used for dataset generation
import matplotlib.pyplot as plt  # Used for plotting the graphs
import matplotlib.animation as animation  # Used for making animation of the graph

# Implementation of Linear Regression Model with Gradient Descent
class Linear_Regression:
    def __init__(self, X, y, learning_rate=0.000001, epoches=500):
        '''
        Initialize the Linear_Regression Model

        Parameters:
        X (numpy array): Input data
        y (numpy array): Target data
        learning_rate (float): Learning rate for gradient descent (default 0.000001)
        epoches (int): Number of epochs for training (default 500)
        '''
        self.X = X
        self.y = y
        self.w = np.random.rand()  # Randomly initialize weight
        self.b = np.random.rand()  # Randomly initialize bias
        self.learning_rate = learning_rate
        self.epoches = epoches

    def accumulated_error(self, y, y_pred):
        '''
        Calculate the accumulated errors for bias and weights
        
        Parameters:
        y (numpy array): True values
        y_pred (numpy array): Predicted values
        
        Returns:
        tuple: Bias error and weight error
        '''
        bias_error = np.mean(y - y_pred)
        weight_error = np.mean((y - y_pred) * self.X)  
        return (bias_error, weight_error)
    
    def train_regressor(self):
        '''
        Train the linear regressor using gradient descent optimization
        
        Returns:
        list: Mean square error for each epoch
        '''
        mse_of_epoches = []
        for epoch in range(self.epoches):
            y_pred = self.w * self.X + self.b  # Predict values
            bias_error, weight_error = self.accumulated_error(self.y, y_pred)  # Calculate errors
            self.b += self.learning_rate * bias_error  # Update bias
            self.w += self.learning_rate * weight_error  # Update weight
            mse = np.mean((self.y - y_pred) ** 2)  # Calculate mean square error
            mse_of_epoches.append(mse)  # Store mean square error

        return mse_of_epoches


# Generate synthetic data
np.random.seed(0)        
x_values = np.random.rand(20) * 10  # Generate random x values
w = 2  # True weight
b = 3  # True bias
noise = np.random.randn(20) * (0.1 * (w * x_values + b))  # Generate noise
y_values = w * x_values + b + noise  # Generate y values with noise

# Create and train the model
model = Linear_Regression(x_values, y_values)
mse_of_epoches = model.train_regressor()

# Print Mean Square Error for each epoch
print("Mean Square Error for each epoch")
for epoch, mse in enumerate(mse_of_epoches):
    print(f"Epoch {epoch}: MSE={mse}")


def animate(i):
    '''
    Animation function to update the plot for each frame
    
    Parameters:
    i (int): Current frame index
    '''
    plt.cla()  # Clear the current axes
    plt.scatter(x_values, y_values, color='black', label='Data Points')  # Plot data points
    plt.plot(x_values, 2 * x_values + 3, color='lightgreen', label='Ground Truth Line')  # Plot ground truth line
    plt.plot(x_values, model.w * x_values + model.b, color='orange', label='Regression Line with Gradient Descent')  # Plot regression line
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Regression Training (Epoch {i+1}/{model.epoches})')
    plt.legend()
    model.train_regressor()  # Train the model
    
# Create figure for animation
fig = plt.figure(figsize=(10, 6))

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=model.epoches, repeat=False)

# Show the animation
plt.show()
