import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load and prepare the dataset
print("--- 1. Loading and Preparing Data ---")
iris = load_iris()
X, y = iris.data, iris.target
print("Dataset loaded successfully.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# Scale the features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features scaled.")

# Convert data to PyTorch tensors
print("Converting data to PyTorch Tensors...")
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
print("Data converted to Tensors.\n")

# 2. Define the neural network model
print("--- 2. Defining the Neural Network ---")
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features, 16 hidden units
        self.fc2 = nn.Linear(16, 3)   # 16 hidden units, 3 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisNet()
print("Model defined:")
print(model)
print("\n")

# 3. Train the model
print("--- 3. Training the Model ---")
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print(f"Using {type(criterion).__name__} as loss function.")
print(f"Using {type(optimizer).__name__} as optimizer.\n")

print("Starting training...")
for epoch in range(200):
    # Forward pass and loss
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')
print("Training complete.\n")

# 4. Evaluate the model
print("--- 4. Evaluating the Model ---")
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%\n')

# 5. Make a prediction
print("--- 5. Making a Prediction on a Sample ---")
# Example: take a sample from the test set
sample = X_test[0].reshape(1, -1)
with torch.no_grad():
    prediction = model(sample)
    _, predicted_class = torch.max(prediction, 1)

    # We need to inverse_transform the scaled sample to see the original features
    original_sample_features = scaler.inverse_transform(sample.numpy())

    print(f'Prediction for a sample flower:')
    print(f'  Original Features: {original_sample_features[0]}')
    print(f'  Predicted class index: {predicted_class.item()}')
    print(f'  Predicted class name: {iris.target_names[predicted_class.item()]}')
    print(f'  Actual class name: {iris.target_names[y_test[0].item()]}')