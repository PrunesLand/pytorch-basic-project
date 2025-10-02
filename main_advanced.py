import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import optuna

# --- 1. Argument Parser ---
def setup_parser():
    parser = argparse.ArgumentParser(description='Advanced Iris Classification with PyTorch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials for hyperparameter search')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--plot_cm', action='store_true', help='Plot confusion matrix for the best model')
    return parser.parse_args()

# --- 2. Custom Dataset ---
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. Advanced Neural Network with Dropout ---
class AdvancedIrisNet(nn.Module):
    def __init__(self, n_layers, dropout_rate, layer_units):
        super(AdvancedIrisNet, self).__init__()

        layers = []
        in_features = 4
        for i in range(n_layers):
            out_features = layer_units[i]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- 4. Training and Evaluation ---
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss

# --- 5. Optuna Objective Function for Hyperparameter Tuning ---
def objective(trial, args, X_train_val, y_train_val, scaler):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    layer_units = []
    for i in range(n_layers):
        layer_units.append(trial.suggest_int(f'n_units_l{i}', 16, 128))

    # K-fold Cross-validation
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_scores = []

    for train_index, val_index in skf.split(X_train_val, y_train_val):
        # Data splitting and scaling for the current fold
        X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
        y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]

        # Create a new scaler for each fold to avoid data leakage
        fold_scaler = StandardScaler()
        X_train_fold = fold_scaler.fit_transform(X_train_fold)
        X_val_fold = fold_scaler.transform(X_val_fold)

        X_train_fold = torch.FloatTensor(X_train_fold)
        y_train_fold = torch.LongTensor(y_train_fold)
        X_val_fold = torch.FloatTensor(X_val_fold)
        y_val_fold = torch.LongTensor(y_val_fold)

        train_dataset = IrisDataset(X_train_fold, y_train_fold)
        val_dataset = IrisDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Create model and optimizer
        model = AdvancedIrisNet(n_layers, dropout_rate, layer_units)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, args.epochs)
        fold_scores.append(val_loss)

    return np.mean(fold_scores)

# --- 6. Confusion Matrix Plotting ---
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

# --- Main Execution ---
def main():
    args = setup_parser()
    print("--- Advanced Iris Classification ---")
    print(f"Running with arguments: {args}\n")

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into a training/validation set and a final test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Hyperparameter Tuning with Optuna ---
    print(f"--- Starting Hyperparameter Tuning ({args.n_trials} trials) ---")
    scaler = StandardScaler()
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args, X_train_val, y_train_val, scaler), n_trials=args.n_trials)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best trial found: {study.best_trial.params}")
    print(f"Best validation loss: {study.best_value:.4f}\n")

    # --- Train Final Model with Best Hyperparameters ---
    print("--- Training Final Model on Full Training Data ---")
    best_params = study.best_trial.params
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_val_tensor = torch.FloatTensor(X_train_val_scaled)
    y_train_val_tensor = torch.LongTensor(y_train_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = IrisDataset(X_train_val_tensor, y_train_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Extract layer units from best parameters
    layer_units = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]

    final_model = AdvancedIrisNet(
        n_layers=best_params['n_layers'],
        dropout_rate=best_params['dropout_rate'],
        layer_units=layer_units
    )

    optimizer = getattr(optim, best_params['optimizer'])(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    # Train on the full training+validation set
    for epoch in range(args.epochs):
        final_model.train()
        for features, labels in train_loader:
            outputs = final_model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')
    print("Final model training complete.\n")

    # --- Evaluate on Test Set ---
    print("--- Evaluating Final Model on Test Set ---")
    final_model.eval()
    with torch.no_grad():
        outputs = final_model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Accuracy on the test set: {accuracy * 100:.2f}%')

        if args.plot_cm:
            cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
            plot_confusion_matrix(cm, classes=iris.target_names)

if __name__ == '__main__':
    main()