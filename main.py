import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_utils import compute_cost, compute_gradient, gradient_descent, predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from visual_utils import visualize_data, visualize_decision_boundary, plot_cost_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from regularize_utils import compute_cost_reg, compute_gradient_reg
df = pd.read_csv("heart.csv")  
df.head()

# Feature Engineering: One-hot encode 
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

# Split features and target
X = df.drop('target', axis=1)
y = df['target'].values.reshape(-1, 1)

# Normalize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept (bias) term

X_processed = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
visualize_data(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


# Train using Gradient Descent

np.random.seed(1)
initial_w = np.random.rand(X_train.shape[1]) - 0.5
initial_b = 1.

# Run gradient descent (non-regularized)
w1, b1, cost_history1, _ = gradient_descent(
    X_train, y_train.ravel(),
    w_in=initial_w,
    b_in=initial_b,
    cost_function=compute_cost,
    gradient_function=compute_gradient,
    alpha=0.1,
    num_iters=1000,
    lambda_=0     # No regularization
)
y_pred_train_1 = predict(X_train, w1, b1)
y_pred_test_1 = predict(X_test, w1, b1)

train_acc1 = np.mean(y_pred_train_1 == y_train.ravel()) * 100
test_acc1 = np.mean(y_pred_test_1 == y_test.ravel()) * 100

print(f"Non-Regularized Logistic Regression")
print(f"Training Accuracy: {train_acc1:.2f}%")
print(f"Test Accuracy: {test_acc1:.2f}%")
print(f"Optimal weights: {w1}")
print(f"Optimal bias: {b1}")
#now well apply regularization to this logistic regression model


# === Train REGULARIZED Logistic Regression ===
np.random.seed(1)
initial_w = np.random.rand(X_train.shape[1]) - 0.5
initial_b = 1.
lambda_ = 0.01

w2, b2, cost_history2,_ = gradient_descent(X_train, y_train,
    w_in=initial_w,
    b_in=initial_b,
    cost_function=compute_cost_reg,
    gradient_function=compute_gradient_reg,
    alpha=0.01,
    num_iters=700,
    lambda_=lambda_)
# Evaluate
y_pred_train_2 = predict(X_train, w2, b2)
y_pred_test_2 = predict(X_test, w2, b2)

train_acc2 = np.mean(y_pred_train_2 == y_train.ravel()) * 100
test_acc2 = np.mean(y_pred_test_2 == y_test.ravel()) * 100
print(f"\nRegularized Logistic Regression")
print(f"Training Accuracy: {train_acc2:.2f}%")  
print(f"Test Accuracy: {test_acc2:.2f}%")
print(f"Optimal weights: {w2}")
print(f"Optimal bias: {b2}")

plot_cost_history(cost_history2)
print("Decision Boundary: Non-Regularized")
visualize_decision_boundary(X_train, y_train, w1, b1)

print("Decision Boundary: Regularized")
visualize_decision_boundary(X_train, y_train, w2, b2)

plt.plot(cost_history1, label="Non-Regularized")
plt.plot(cost_history2, label="Regularized (λ=0.01)")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.legend()
plt.grid(True)
plt.show()