import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def smooth_labels(targets, smoothing=0.1):
    return targets * (1 - smoothing) + 0.5 * smoothing

def evaluate_model(y, y_pred):
    acc = accuracy_score(y, (y_pred > 0.5).to(T.int))

    # Calculate F1 Score (binary classification)
    f1 = f1_score(y, (y_pred > 0.5).to(T.int))
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y, y_pred)
        print(f"AUC {auc:.4f}")
    except AttributeError:
        print("It doesn't return probabilty")

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to tensors
        X_tensor = self.X[idx].to(T.float)
        y_tensor = self.y[idx].to(T.float)
        return X_tensor, y_tensor

class ResNetLayer(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout_rate: float = 0.):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, input):
        output = T.add(input, self.mlp(input))
        output = self.norm(output)
        return output
    
class AttentionLikeLayer(nn.Module):
    def __init__(self, input_dim: int, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.softmax_layer = nn.Linear(input_dim, dim * num_heads)
        self.normal_layer = nn.Linear(input_dim, dim * num_heads)
        self.output_layer = nn.Linear(dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, input):
        softmax_layer = self.softmax_layer(input)
        softmax_layer = softmax_layer.view(-1, self.num_heads, self.dim)
        softmax_layer = F.softmax(softmax_layer, dim=-1)
        
        normal_layer = self.normal_layer(input)
        normal_layer = normal_layer.view(-1, self.num_heads, self.dim)
        
        output = T.multiply(softmax_layer, normal_layer)
        output = output.sum(dim=1)
        output = self.output_layer(output)
        output = T.add(output, input)
        output = self.norm(output)
        return output
        

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, mult: int = 4, dropout_rate: float = 0.):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            AttentionLikeLayer(hidden_dims, hidden_dims * mult, 2),
            ResNetLayer(hidden_dims, dropout_rate=dropout_rate, mult=mult),
            AttentionLikeLayer(hidden_dims, hidden_dims * mult, 2),
            ResNetLayer(hidden_dims, dropout_rate=dropout_rate, mult=mult),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Linear(hidden_dims, 1)
        )
    def forward(self, input):
        output = self.mlp(input)
        return output
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' for loss, 'max' for accuracy, etc.
        self.state_dict = None

    def __call__(self, current_score, state_dict):
        if self.best_score is None:
            self.best_score = current_score
        elif (
            (self.mode == 'min' and current_score > self.best_score - self.min_delta) or
            (self.mode == 'max' and current_score < self.best_score + self.min_delta)
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
            self.state_dict = state_dict
            print(f"New best score {current_score:.6f}")
            
            
class CombLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gelu_layer = nn.Linear(dim, dim)
        self.exp_layer = nn.Linear(dim, dim)
        self.final_layer = nn.Linear(2 * dim, dim)
        
    def forward(self, input):
        gelu = F.gelu(self.gelu_layer(input))
        exp = T.exp(self.exp_layer(input)).clip(max=100)
        x = T.concat([gelu, exp], dim=-1)
        return self.final_layer(x)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            # CombLayer(hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims),
            # CombLayer(hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1)
        )
    def forward(self, input):
        output = self.mlp(input)
        return output