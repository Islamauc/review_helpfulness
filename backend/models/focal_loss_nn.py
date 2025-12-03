"""
Neural Network with Focal Loss and Class-Weighted BCE
Implements a PyTorch-based neural network that supports both focal loss
and class-weighted binary cross entropy loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional, Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability for the true class.
    
    This loss down-weights easy examples and focuses on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for the rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean' or 'sum' (default: 'mean')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Raw logits from the model (batch_size,)
            targets: True labels (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute p_t: probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE component
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Class-weighted BCE + Focal Loss
    
    This combines the benefits of both:
    - Class-weighted BCE: Handles class imbalance
    - Focal Loss: Focuses on hard examples
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        bce_weight: float = 0.5
    ):
        """
        Args:
            class_weights: Tensor of shape (2,) with weights for [class_0, class_1]
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            focal_weight: Weight for focal loss component (default: 0.5)
            bce_weight: Weight for class-weighted BCE component (default: 0.5)
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.class_weights = class_weights
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Raw logits from the model (batch_size,)
            targets: True labels (batch_size,)
        
        Returns:
            Combined loss value
        """
        # Focal loss component
        focal = self.focal_loss(inputs, targets)
        
        # Class-weighted BCE component
        if self.class_weights is not None:
            # Create weight tensor for each sample
            weights = self.class_weights[targets.long()]
            bce = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, weight=weights, reduction='mean'
            )
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='mean'
            )
        
        # Combine losses
        combined = self.focal_weight * focal + self.bce_weight * bce
        
        return combined


class TabularDataset(Dataset):
    """Dataset class for tabular data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLossNN(nn.Module):
    """Neural Network architecture for binary classification."""
    
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (100,)):
        """
        Args:
            input_dim: Number of input features
            hidden_sizes: Tuple of hidden layer sizes (default: (100,))
        """
        super(FocalLossNN, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # Output layer (single neuron for binary classification)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze()


class FocalLossClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier with focal loss and class-weighted BCE.
    
    This class provides a sklearn-like interface while using PyTorch internally
    for training with custom loss functions.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        learning_rate: float = 0.001,
        max_iter: int = 100,
        batch_size: int = 256,
        random_state: Optional[int] = None,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        bce_weight: float = 0.5,
        class_weight: Optional[str] = 'balanced',
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        verbose: bool = False
    ):
        """
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes (default: (100,))
            learning_rate: Learning rate for optimizer (default: 0.001)
            max_iter: Maximum number of epochs (default: 100)
            batch_size: Batch size for training (default: 256)
            random_state: Random seed for reproducibility
            use_focal_loss: Whether to use focal loss (default: True)
            focal_alpha: Alpha parameter for focal loss (default: 0.25)
            focal_gamma: Gamma parameter for focal loss (default: 2.0)
            focal_weight: Weight for focal loss component (default: 0.5)
            bce_weight: Weight for class-weighted BCE component (default: 0.5)
            class_weight: 'balanced' or None for class weights (default: 'balanced')
            early_stopping: Whether to use early stopping (default: True)
            validation_fraction: Fraction of data to use for validation (default: 0.1)
            n_iter_no_change: Number of iterations with no improvement for early stopping (default: 10)
            verbose: Whether to print training progress (default: False)
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        
        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None
    
    def _compute_class_weights(self, y: np.ndarray) -> Optional[torch.Tensor]:
        """Compute class weights from training data."""
        if self.class_weight is None:
            return None
        
        if self.class_weight == 'balanced':
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            # Create a mapping: class -> weight
            weight_dict = dict(zip(classes, weights))
            # Return tensor with weights for [class_0, class_1]
            weight_tensor = torch.FloatTensor([
                weight_dict.get(0, 1.0),
                weight_dict.get(1, 1.0)
            ])
            return weight_tensor
        
        return None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FocalLossClassifier':
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        # Set random seeds for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
        
        # Store metadata
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Convert to binary classification if needed
        if len(self.classes_) > 2:
            raise ValueError("This classifier only supports binary classification")
        
        # Compute class weights
        class_weights = self._compute_class_weights(y)
        
        # Split data for validation if early stopping is enabled
        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Create datasets
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Initialize model
        self.model_ = FocalLossNN(
            input_dim=self.n_features_in_,
            hidden_sizes=self.hidden_layer_sizes
        )
        
        # Initialize loss function
        if self.use_focal_loss:
            loss_fn = CombinedLoss(
                class_weights=class_weights,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                focal_weight=self.focal_weight,
                bce_weight=self.bce_weight
            )
        else:
            # Use only class-weighted BCE
            if class_weights is not None:
                loss_fn = nn.BCEWithLogitsLoss(
                    pos_weight=class_weights[1] / class_weights[0]
                )
            else:
                loss_fn = nn.BCEWithLogitsLoss()
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(self.max_iter):
            # Training phase
            self.model_.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            if self.early_stopping:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model_(batch_X)
                        loss = loss_fn(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    # Save best model state
                    self.best_model_state_ = self.model_.state_dict().copy()
                else:
                    no_improve_count += 1
                
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.max_iter} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if no_improve_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    self.model_.load_state_dict(self.best_model_state_)
                    break
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.max_iter} - Train Loss: {train_loss:.4f}")
        
        self.model_.eval()
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples, 2) for [class_0, class_1]
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model_.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.sigmoid(logits)
        
        # Convert to numpy and format as (n_samples, 2)
        probs_np = probs.numpy()
        return np.column_stack([1 - probs_np, probs_np])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def __getstate__(self):
        """Custom serialization for joblib."""
        state = self.__dict__.copy()
        if 'model_' in state and state['model_'] is not None:
            state['model_state_dict_'] = state['model_'].state_dict()
            del state['model_']
        return state
    
    def __setstate__(self, state):
        """Custom deserialization for joblib."""
        if 'model_state_dict_' in state:
            # Reconstruct model
            model = FocalLossNN(
                input_dim=state['n_features_in_'],
                hidden_sizes=state['hidden_layer_sizes']
            )
            model.load_state_dict(state['model_state_dict_'])
            state['model_'] = model
            del state['model_state_dict_']
        self.__dict__.update(state)

