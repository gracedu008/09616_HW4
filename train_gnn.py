import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_add_pool, MessagePassing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
import time
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import add_safe_globals

# Set paths for the server environment
data_dir = "/home/gdu/09616_HW4/data"
output_dir = "/home/gdu/09616_HW4"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Dataset class - already provided in base code
class QM_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(root=".")
        # Add this line to allow GlobalStorage to be unpickled
        add_safe_globals([GlobalStorage])
        # Now load the data
        self.data = torch.load(path)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

# Load the datasets
train_path = os.path.join(data_dir, "train.pt")
test_path = os.path.join(data_dir, "test.pt")

print(f"Loading data from {train_path} and {test_path}")
train_data_ = QM_Dataset(train_path)
test_data = QM_Dataset(test_path)

# Print dataset info
print(f"Train dataset size: {len(train_data_)}")
print(f"Test dataset size: {len(test_data)}")

# Get a sample to examine the structure
sample = train_data_[0]
print(f"Sample data structure: {sample}")
print(f"Node features shape: {sample.x.shape}")
if hasattr(sample, 'edge_attr'):
    print(f"Edge features shape: {sample.edge_attr.shape}")
print(f"Edge index shape: {sample.edge_index.shape}")

# Split the train dataset for validation
train_size = int(len(train_data_) * 0.95)
val_size = len(train_data_) - train_size
train_data, validate_data = torch.utils.data.random_split(train_data_, [train_size, val_size])

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validate_data)}")

# GNN Model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features=11, hidden_dim=128):
        super(GNN, self).__init__()
        
        # Node embedding layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*2)
        self.bn3 = BatchNorm1d(hidden_dim*2)
        
        # Readout layers (graph level)
        self.fc1 = Linear(hidden_dim*2, hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim//2)
        self.bn5 = BatchNorm1d(hidden_dim//2)
        self.fc3 = Linear(hidden_dim//2, 1)
        
        # Dropout for regularization
        self.dropout = Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embedding
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Readout (graph-level pooling)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim*2]
        
        # Prediction MLP
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x.view(-1)

# Define a more advanced model - GIN (Graph Isomorphism Network)
class GINNet(torch.nn.Module):
    def __init__(self, num_node_features=11, hidden_dim=128):
        super(GINNet, self).__init__()
        
        # GIN layers
        nn1 = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.conv1 = GINConv(nn1)
        
        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim*2),
            BatchNorm1d(hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, hidden_dim*2),
            BatchNorm1d(hidden_dim*2),
            ReLU()
        )
        self.conv2 = GINConv(nn2)
        
        nn3 = Sequential(
            Linear(hidden_dim*2, hidden_dim*2),
            BatchNorm1d(hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, hidden_dim*2),
            BatchNorm1d(hidden_dim*2),
            ReLU()
        )
        self.conv3 = GINConv(nn3)
        
        # Output layers
        self.fc1 = Linear(hidden_dim*2, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embedding with GIN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Prediction MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x.view(-1)

# Training function
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.l1_loss(out, data.y)  # L1 Loss (MAE)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    mae = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            mae += F.l1_loss(out, data.y, reduction='sum').item()
    
    return mae / len(loader.dataset)

# Prediction function
def predict(model, loader, device):
    model.eval()
    predictions = []
    indices = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            predictions.extend(out.cpu().numpy())
            indices.extend([d.name for d in data.to_data_list()])
    
    return indices, predictions

# Set batch sizes and create data loaders
# Try a larger batch size for H100 GPU
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(validate_data, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model, optimizer, and scheduler
model = GINNet().to(device)  # Use GINNet for better performance
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)

# Training loop
num_epochs = 300  # Increase epochs since we have a powerful GPU
best_val_mae = float('inf')
best_model_state = None
train_losses = []
val_losses = []
patience = 20  # Increase patience a bit
counter = 0

# Add a timestamp for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_save_path = os.path.join(output_dir, f"best_model_{timestamp}.pt")
results_save_path = os.path.join(output_dir, f"training_results_{timestamp}.txt")
plot_save_path = os.path.join(output_dir, f"training_curve_{timestamp}.png")
submission_save_path = os.path.join(output_dir, f"submission_{timestamp}.csv")

# Open a file to log results
with open(results_save_path, 'w') as f:
    f.write(f"Training started at {timestamp}\n")
    f.write(f"Model: GINNet\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
    f.write(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}\n")
    f.write(f"Device: {device}\n\n")
    f.write("Epoch,Train Loss,Val MAE\n")

print(f"Starting training at {timestamp}...")
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, device)
    val_mae = evaluate(model, val_loader, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_mae)
    
    # Learning rate scheduling
    scheduler.step(val_mae)
    
    # Print progress
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')
    
    # Log results
    with open(results_save_path, 'a') as f:
        f.write(f"{epoch},{train_loss:.6f},{val_mae:.6f}\n")
    
    # Save best model
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, model_save_path)
        print(f"New best model saved with validation MAE: {best_val_mae:.4f}")
        counter = 0
    else:
        counter += 1
    
    # Early stopping
    if counter >= patience:
        print(f'Early stopping after {epoch} epochs')
        break

print(f'Best validation MAE: {best_val_mae:.4f}')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(plot_save_path)
plt.close()

# Load best model for prediction
model.load_state_dict(torch.load(model_save_path))

# Make predictions on test set
print("Making predictions on test set...")
indices, predictions = predict(model, test_loader, device)

# Create submission dataframe
submission_df = pd.DataFrame({
    'Idx': indices,
    'labels': predictions
})

# Save submission to CSV
submission_df.to_csv(submission_save_path, index=False)
print(f"Submission file created at {submission_save_path}!")

# Also save a standard name for easy reference
standard_submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(standard_submission_path, index=False)
print(f"Standard submission file created at {standard_submission_path}!")

print("Training and prediction complete!")