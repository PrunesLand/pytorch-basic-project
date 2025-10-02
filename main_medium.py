import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Custom Dataset for Iris ---
print("--- 1. Defining Custom Iris Dataset ---")
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
print("Custom IrisDataset class defined.\n")

# --- 2. Loading and Preparing Data ---
print("--- 2. Loading and Preparing Data ---")
iris = load_iris()
X, y = iris.data, iris.target
print("Dataset loaded successfully.")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Data split into {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} testing samples.")

# Scale the features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print("Features scaled.")

# Convert to PyTorch Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)
print("Data converted to Tensors.")

# Create Dataset and DataLoader instances
train_dataset = IrisDataset(X_train, y_train)
val_dataset = IrisDataset(X_val, y_val)
test_dataset = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("DataLoaders created with batch size 16.\n")

# --- 3. Defining a More Complex Neural Network ---
print("--- 3. Defining a More Complex Neural Network ---")
class EnhancedIrisNet(nn.Module):
    def __init__(self):
        super(EnhancedIrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = EnhancedIrisNet()
print("Model defined:")
print(model)
print("\n")

# --- 4. Training the Model with Validation ---
print("--- 4. Training the Model with Validation ---")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
print(f"Using {type(criterion).__name__} and {type(optimizer).__name__}.\n")

best_val_loss = float('inf')
epochs = 150
print(f"Starting training for {epochs} epochs...")
for epoch in range(epochs):
    model.train()  # Set model to training mode
    for features, labels in train_loader:
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}')
print("Training complete.\n")

# --- 5. Evaluating the Best Model on the Test Set ---
print("--- 5. Evaluating the Best Model on the Test Set ---")
# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the best model on the test set: {accuracy * 100:.2f}%\n')

# --- 6. Making a Prediction with the Loaded Model ---
print("--- 6. Making a Prediction ---")
sample_features, sample_label = test_dataset[0]
sample_features = sample_features.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    prediction = model(sample_features)
    _, predicted_class = torch.max(prediction, 1)

    original_sample_features = scaler.inverse_transform(sample_features.numpy())

    print(f'Prediction for a sample flower:')
    print(f'  Original Features: {original_sample_features[0]}')
    print(f'  Predicted class: {iris.target_names[predicted_class.item()]}')
    print(f'  Actual class: {iris.target_names[sample_label]}')