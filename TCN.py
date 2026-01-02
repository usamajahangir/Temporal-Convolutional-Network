import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import urllib.request
import zipfile
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============== Config Values ==========================================

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
DATASET_DIR = 'UCI HAR Dataset'
DATASET_ZIP = 'UCI_HAR_Dataset.zip'

NUM_CLASSES = 6

INPUT_CHANNELS = 9
INITIAL_CHANNELS = 64
EXPANDED_CHANNELS = 128
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8]
DROPOUT_TCN = 0.5
DROPOUT_CLASSIFIER = 0.5

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

# OVERFIT_THRESHOLD = 15.0
EARLY_STOP_PATIENCE = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_PATH = 'tcn_har_model.pth'
TRAINING_PLOT_PATH = 'training_curves.png'

# ====================  LOGGING SETUP =======================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================  DATA LOADING AND PREPROCESSING  =============================

def download_and_extract_dataset():
    """Download if dataset not available and extract UCI HAR dataset"""
    
    if os.path.exists(DATASET_DIR):
        logger.info(f"Dataset already exists at {DATASET_DIR}")
        return DATASET_DIR
    
    logger.info("Downloading UCI HAR dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
    logger.info("Download complete. Extracting...")
    
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    os.remove(DATASET_ZIP)
    logger.info(f"Dataset extracted to UCI HAR Dataset")
    return 'UCI HAR Dataset'


def load_signals(signal_dir: str, signal_types: List[str], data_type: str) -> np.ndarray:
    """
    Load multiple signal and contatinate them
    Args:
        signal_dir: Directory path containing signal files
        signal_types: List of signal names
        data_type: 'train' or 'test' to determine file suffix
    """
    signals = []
    
    if not os.path.exists(signal_dir):
        raise FileNotFoundError(f"Signal directory does not exist: {signal_dir}")
    
    for signal_type in signal_types:
        filename = os.path.join(signal_dir, f'{signal_type}_{data_type}.txt')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing file: {filename}")
        
        if filename is None:
            raise FileNotFoundError(f"Could not find signal file for '{signal_type}' in {signal_dir}. ")
        
        data = np.loadtxt(filename)
        signals.append(data)
        logger.debug(f"Loaded {os.path.basename(filename)} with shape {data.shape}")

    stacked = np.stack(signals, axis=-1)
    logger.info(f"Stacked signals shape: {stacked.shape}")
    return stacked


def load_har_data(data_type: str, dataset_base_path: str):
    """
    Load UCI HAR dataset (train or test)
    Args:
        data_type: 'train' or 'test'
        dataset_base_path: path where data is located
    Returns:
        X: Shape (num_samples, sequence_length, num_channels)
        y: Shape (num_samples,)
    """
    
    base_dir = os.path.join(dataset_base_path, data_type)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Could not find {data_type} directory in {dataset_base_path}")

    logger.info(f"Found {data_type} data at: {base_dir}")
    
    if base_dir is None:
        raise FileNotFoundError(f"Could not find {data_type} directory in {dataset_base_path}")
    
    signal_dir = os.path.join(base_dir, 'Inertial Signals')
    
    if not os.path.exists(signal_dir):
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")
    
    signal_types = [ 'body_acc_x', 'body_acc_y', 'body_acc_z',
                     'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                     'total_acc_x', 'total_acc_y', 'total_acc_z' ]
    
    X = load_signals(signal_dir, signal_types, data_type)
    
    labels_file = os.path.join(base_dir, f'y_{data_type}.txt')
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Could not find labels file at {labels_file}")

    y = np.loadtxt(labels_file, dtype=int) - 1
    logger.info(f"Loaded {data_type} data: X shape {X.shape}, y shape {y.shape}")
    return X, y


def compute_normalization_stats(X_train: np.ndarray):
    """
    Compute mean and std for z-score normalization
    Args:
        X_train: Shape (num_samples, sequence_length, num_channels)
    Returns:
        mean: Shape (num_channels,)
        std: Shape (num_channels,)
    """
    
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    logger.info(f"Normalization stats - Mean: {mean}, Std: {std}")
    return mean, std


def normalize_data(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Apply z-score normalization
    Pre-Req: compute_normalization_stats
    """
    return (X - mean) / std


# ===================== Dataset Class ======================================

class HARDataset(Dataset):
    """Human Activity Recognition Dataset"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Shape (num_samples, sequence_length, num_channels)
            y: Shape (num_samples,)
        """
        self.X = torch.FloatTensor(X).permute(0, 2, 1)
        self.y = torch.LongTensor(y) 
    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# ==================== Model Arch  ========================================

class CausalConv1d(nn.Module):
    """Causal convolution with weight normalization"""
    
    def __init__(self, channel_inannels: int, channel_outannels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channel_inannels, channel_outannels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv = nn.utils.weight_norm(self.conv)
    
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block with residual connection"""
    
    def __init__(self, channel_inannels: int, channel_outannels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        
        self.conv = CausalConv1d(channel_inannels, channel_outannels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(channel_inannels, channel_outannels, 1) if channel_inannels != channel_outannels else None
        if self.residual is not None:
            self.residual = nn.utils.weight_norm(self.residual)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        residual = x if self.residual is None else self.residual(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence classification"""
    
    def __init__(self, input_channels: int, num_classes: int, initial_channels: int,
        expanded_channels: int, kernel_size: int, dilations: List[int], dropout_tcn: float, dropout_classifier: float):
        
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, initial_channels, kernel_size=1)

        tcn_blocks = []
        channel_in = initial_channels
        
        for i, dilation in enumerate(dilations):
            channel_out = expanded_channels if i >= len(dilations) // 2 else initial_channels
            tcn_blocks.append(TCNBlock(channel_in, channel_out, kernel_size, dilation, dropout_tcn))
            channel_in = channel_out
        
        self.tcn_blocks = nn.ModuleList(tcn_blocks)
        self.classifier = nn.Sequential(nn.Dropout(dropout_classifier), nn.Linear(channel_in, num_classes))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_channels, sequence_length)
        Returns:
            output: (batch_size, num_classes)
        """

        x = self.input_projection(x)
        for block in self.tcn_blocks:
            x = block(x)
        
        x = x.mean(dim=2)
        
        output = self.classifier(x)
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===================== Helpr func for training  ============================

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ===================== Traning function/loop ==============================

def train_model():
    """Main training function"""
    
    dataset_path = download_and_extract_dataset()
    
    logger.info("Loading data...")
    X_train, y_train = load_har_data('train', dataset_path)
    X_test, y_test = load_har_data('test', dataset_path)
    
    num_val = int(0.30 * len(X_train))
    X_val = X_train[-num_val:]
    y_val = y_train[-num_val:]
    X_train = X_train[:-num_val]
    y_train = y_train[:-num_val]
    
    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    mean, std = compute_normalization_stats(X_train)
    X_train = normalize_data(X_train, mean, std)
    X_val = normalize_data(X_val, mean, std)
    X_test = normalize_data(X_test, mean, std)
    
    train_dataset = HARDataset(X_train, y_train)
    val_dataset = HARDataset(X_val, y_val)
    test_dataset = HARDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    logger.info("Creating model...")
    model = TemporalConvNet(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        initial_channels=INITIAL_CHANNELS,
        expanded_channels=EXPANDED_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dilations=DILATIONS,
        dropout_tcn=DROPOUT_TCN,
        dropout_classifier=DROPOUT_CLASSIFIER
    )
    
    model = model.to(DEVICE)
    
    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Device: {DEVICE}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    logger.info("Starting training...\n" + "=" * 70)
    
    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        logger.info( f"Epoch [{epoch+1}/{NUM_EPOCHS}] | " 
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)      
        overfit_gap = train_acc - val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'overfit_gap': overfit_gap
            }, MODEL_SAVE_PATH)
            logger.info(f"✓ Saved best model | Val Acc: {val_acc:.2f}% | Gap: {overfit_gap:.2f}%")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # if overfit_gap > OVERFIT_THRESHOLD:
        #     logger.info(f"Early stopping triggered at epoch {epoch+1} due to overfitting (gap: {overfit_gap:.2f}% > {OVERFIT_THRESHOLD}%)")
        #     break

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    logger.info("=" * 70 + "\nTraining complete!")
    
    logger.info(f"Loading best model (val_acc: {best_val_acc:.2f}%)...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")

    plot_training_curves(history)
    return model, history


# ==================== Plotting =======================================

def plot_training_curves(history: dict):
    """func to plot the results and save images"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2.5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2.5)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12) 
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    fig.suptitle('Model Training Curves', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(TRAINING_PLOT_PATH, dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to {TRAINING_PLOT_PATH}")
    plt.close()

def main():
    logger.info("=" * 70 + "\nTASK 2: TEMPORAL CONVOLUTIONAL NETWORK FOR HAR\n" + "=" * 70)
    logger.info(f"Dataset: UCI Human Activity Recognition")
    logger.info(f"Model: TCN with dilated causal convolutions")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Device: {DEVICE}\n" + "=" * 70)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Kindly verify GPU device.")
    else:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            
    model, history = train_model()
    
    logger.info("=" * 70 + "\nComplted!\n")
    logger.info(f"Model file saved to: {MODEL_SAVE_PATH}")
    logger.info(f"Training graph images saved to: {TRAINING_PLOT_PATH}\n" + "=" * 70)


if __name__ == "__main__":
    main()