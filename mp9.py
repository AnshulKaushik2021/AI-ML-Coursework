import numpy as np
from planar_arm import Arm

def create_linear_data(num_samples, slope, intercept, x_range=[-1.0, 1.0], noise=0.1):
    x = np.random.uniform(x_range[0], x_range[1], (num_samples, 1))
    y = slope * x + intercept + np.random.uniform(-noise, noise, (num_samples, 1))
    return x, y

def get_simple_linear_features(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))

def linear_prediction(x, A, get_modified_features):
    X_modified = get_modified_features(x)
    return X_modified @ A

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def compute_model_error(x, y, A, get_modified_features):
    return mse_loss(linear_prediction(x, A, get_modified_features), y)

def analytical_linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def get_linear_regression_gradient(A, X, y):
    N = X.shape[0]
    return 2 * X.T @ (X @ A - y) / N

def gradient_descent(get_gradient, A_init, learning_rate, num_iterations):
    A = A_init.astype(np.float64)  # Ensure A is float
    for _ in range(num_iterations):
        A -= learning_rate * get_gradient(A)
    return A

def stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate, num_epochs, data_size, batch_size):
    A = A_init.astype(np.float64)  # Ensure A is float
    for _ in range(num_epochs):
        indices = np.random.permutation(data_size)
        for i in range(0, data_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            A -= learning_rate * get_batch_gradient(A, batch_indices)
    return A

def create_sine_data(num_samples, x_range=[0.0, 2 * np.pi], noise=0.1):
    x = np.random.uniform(x_range[0], x_range[1], (num_samples, 1))
    y = np.sin(x) + np.random.uniform(-noise, noise, (num_samples, 1))
    return x, y

def get_polynomial_features(x, degree):
    return np.hstack([x ** (degree - i) for i in range(degree + 1)])

def ik_loss(arm: Arm, config, goal):
    ee_position = arm.forward_kinematics(config)[-1]
    return np.linalg.norm(ee_position - goal)

def ik_loss_with_obstacles(arm: Arm, config, goal, obstacles):
    ee_loss = ik_loss(arm, config, goal)
    workspace_config = arm.forward_kinematics(config)
    total_obstacle_loss = 0
    for obstacle in obstacles:
        obstacle_dist = np.min(np.linalg.norm(workspace_config - obstacle[:2], axis=1))
        if obstacle_dist < obstacle[2]:
            return np.inf
        total_obstacle_loss += 1 / (obstacle_dist - obstacle[2])
    return ee_loss + total_obstacle_loss

def sample_near(num_samples, config, epsilon=0.1):
    return config + np.random.uniform(-epsilon, epsilon, (num_samples, len(config)))

def estimate_ik_gradient(loss, config, num_samples):
    samples = sample_near(num_samples, config)
    losses = np.array([loss(sample) for sample in samples])
    max_loss_idx = np.argmax(losses)
    gradient = samples[max_loss_idx] - config
    return gradient / np.linalg.norm(gradient)

def logistic_error(y_pred, y_true):
    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions != y_true)

def logistic_prediction(x, A, get_modified_features):
    X_modified = get_modified_features(x)
    linear_pred = X_modified @ A
    return 1 / (1 + np.exp(-linear_pred))

def logistic_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def get_logistic_regression_gradient(A, X, y):
    y_pred = logistic_prediction(X, A, lambda x: x)
    return X.T @ (y_pred - y)


def get_logistic_regression_features(x):
    features = [x, np.ones((x.shape[0], 1))]
    
    features.append(x ** 2)
    features.append(x ** 3)
    features.append(x ** 4)
    
    num_features = x.shape[1]
    for i in range(num_features):
        for j in range(i, num_features):
            features.append((x[:, i] * x[:, j]).reshape(-1, 1))
    
    features.append(np.sin(x))
    features.append(np.cos(x))
    
    return np.hstack(features)





