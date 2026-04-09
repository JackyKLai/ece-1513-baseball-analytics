import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_pipeline import prepare_data, _REPO_ROOT, FEATURE_COLS
from evaluate import compute_mae, compute_rmse, plot_pred_vs_actual, print_metrics

DEVICE = torch.device("mps")

HIDDEN1 = 64
HIDDEN2 = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 300
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2
VAL_FRACTION = 0.15
PATIENCE = 20

RESULTS_DIR = os.path.join(_REPO_ROOT, 'results', 'neural_net')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class WinPredictor(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT_RATE):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.drop1  = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.drop2  = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden2, 1)
        self.relu   = nn.ReLU()

    def forward(self, x):
        h1 = self.drop1(self.relu(self.layer1(x)))
        h2 = self.drop2(self.relu(self.layer2(h1)))
        logits = self.output(h2)
        return logits


class WinPredictorOneLayer(nn.Module):
    def __init__(self, input_dim, hidden1, dropout=DROPOUT_RATE):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.drop1  = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden1, 1)
        self.relu   = nn.ReLU()

    def forward(self, x):
        h1     = self.drop1(self.relu(self.layer1(x)))
        logits = self.output(h1)
        return logits


def _to_tensors(X, y):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    return X_t, y_t


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n_samples    = 0
    for X_batch, y_batch in loader:
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(X_batch)
        n_samples    += len(X_batch)
    return running_loss / n_samples


def validate(model, X_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        loss   = criterion(logits, y_val)
        preds  = logits.squeeze().cpu().numpy()
    return loss.item(), preds


def train_model(model, X_train, y_train, X_val=None, y_val=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    model = model.to(DEVICE)
    X_tr, y_tr = _to_tensors(X_train, y_train)
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_v, y_v = _to_tensors(X_val, y_val)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val = float('inf')
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer)
        train_losses.append(train_loss)

        if has_val:
            val_loss, _ = validate(model, X_v, y_v, criterion)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val': best_val,
        'epochs_run': len(train_losses),
    }


def predict(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
    return logits.squeeze().cpu().numpy()


def split_train_val(X_train, y_train, val_fraction=VAL_FRACTION):
    n = len(y_train)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    return (X_train[:n_train], y_train[:n_train], X_train[n_train:], y_train[n_train:])


def run_single(X_train_scaled, y_train, X_test_scaled, y_test, hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT_RATE, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, weight_decay=WEIGHT_DECAY, patience=PATIENCE, val_fraction=VAL_FRACTION, seed=None):
    if seed:
        set_seed(seed)

    input_dim = X_train_scaled.shape[1]
    X_tr, y_tr, X_v, y_v = split_train_val(X_train_scaled, y_train, val_fraction)
    model = WinPredictor(input_dim, hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters())

    history = train_model(model, X_tr, y_tr, X_v, y_v, num_epochs=num_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience)

    preds = predict(history['model'], X_test_scaled)
    mae = compute_mae(y_test, preds)
    rmse = compute_rmse(y_test, preds)

    return {
        'model': history['model'],
        'preds': preds,
        'mae': mae,
        'rmse': rmse,
        'n_params': n_params,
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'best_epoch': history['best_epoch'],
        'epochs_run': history['epochs_run'],
        'config': {
            'hidden1': hidden1, 'hidden2': hidden2, 'dropout': dropout,
            'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay
        }
    }


DEFAULT_SEEDS = [0, 1, 2, 3, 4]

def run_multi_seed(X_train_scaled, y_train, X_test_scaled, y_test, seeds=None, n_seeds=5, **kwargs):
    if not seeds:
        seeds = list(range(n_seeds))

    per_seed = []
    for s in seeds:
        res = run_single(X_train_scaled, y_train, X_test_scaled, y_test, seed=s, **kwargs)
        per_seed.append(res)

    maes  = np.array([r['mae']  for r in per_seed])
    rmses = np.array([r['rmse'] for r in per_seed])

    return {
        'per_seed': per_seed,
        'seeds': seeds,
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'maes': maes,
        'rmses': rmses
    }

def run_sweep(X_train_scaled, y_train, X_test_scaled, y_test, param_name, param_values, param_labels=None, n_seeds=3, base_kwargs=None):
    if not base_kwargs:
        base_kwargs = {}
    if not param_labels:
        param_labels = [str(v) for v in param_values]

    results = {}
    for val, label in zip(param_values, param_labels):
        kw = dict(base_kwargs)
        kw[param_name] = val
        print("  " + param_name + "=" + label + " ...", end=" ", flush=True)
        ms = run_multi_seed(X_train_scaled, y_train, X_test_scaled, y_test, n_seeds=n_seeds, **kw)
        results[label] = {
            'value': val,
            'mae_mean': ms['mae_mean'],
            'mae_std': ms['mae_std'],
            'rmse_mean': ms['rmse_mean'],
            'rmse_std': ms['rmse_std'],
            'maes': ms['maes'],
            'rmses': ms['rmses']
        }
        print("MAE=" + f"{ms['mae_mean']:.2f}" + "+/-" + f"{ms['mae_std']:.2f}  RMSE=" + f"{ms['rmse_mean']:.2f}" + "+/-" + f"{ms['rmse_std']:.2f}")
    return results


WD_VALUES = [0, 1e-5, 1e-4, 1e-3, 1e-2]
WD_LABELS = ['0', '1e-5', '1e-4', '1e-3', '1e-2']
DROP_VALUES = [0, 0.1, 0.2, 0.3, 0.5]
DROP_LABELS = ['0', '0.1', '0.2', '0.3', '0.5']
BS_VALUES = [16, 32, 64]
BS_LABELS = ['16', '32', '64']
LR_VALUES = [3e-4, 1e-3, 3e-3]
LR_LABELS = ['3e-4', '1e-3', '3e-3']


def run_weight_decay_sweep(X_train_scaled, y_train, X_test_scaled, y_test, n_seeds=3):
    return run_sweep(X_train_scaled, y_train, X_test_scaled, y_test, param_name='weight_decay', param_values=WD_VALUES, param_labels=WD_LABELS, n_seeds=n_seeds)


def run_dropout_sweep(X_train_scaled, y_train, X_test_scaled, y_test, n_seeds=3):
    return run_sweep(X_train_scaled, y_train, X_test_scaled, y_test, param_name='dropout', param_values=DROP_VALUES, param_labels=DROP_LABELS, n_seeds=n_seeds)


def run_batch_size_sweep(X_train_scaled, y_train, X_test_scaled, y_test, n_seeds=3):
    return run_sweep(X_train_scaled, y_train, X_test_scaled, y_test, param_name='batch_size', param_values=BS_VALUES, param_labels=BS_LABELS, n_seeds=n_seeds)


def run_learning_rate_sweep(X_train_scaled, y_train, X_test_scaled, y_test, n_seeds=3):
    return run_sweep(X_train_scaled, y_train, X_test_scaled, y_test, param_name='lr', param_values=LR_VALUES, param_labels=LR_LABELS, n_seeds=n_seeds)


# plotting helpers
def plot_training_curve(train_losses, val_losses=None, best_epoch=None, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training loss', linewidth=1.5)
    if val_losses:
        ax.plot(epochs, val_losses, label='Validation loss', linewidth=1.5)
    if best_epoch and val_losses:
        ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.6, label='Early stop (epoch ' + str(best_epoch) + ')')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return ax

def plot_sweep_results(sweep_results, param_name, title_suffix='', save_path=None, show=False):
    labels = list(sweep_results.keys())
    mae_means = [sweep_results[l]['mae_mean']  for l in labels]
    mae_stds = [sweep_results[l]['mae_std']   for l in labels]
    rmse_means = [sweep_results[l]['rmse_mean'] for l in labels]
    rmse_stds = [sweep_results[l]['rmse_std']  for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    ax.bar(x - width / 2, mae_means,  width, yerr=mae_stds,  capsize=4, label='MAE',  alpha=0.8, color='steelblue')
    ax.bar(x + width / 2, rmse_means, width, yerr=rmse_stds, capsize=4, label='RMSE', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Error (wins)')
    ax.set_title(param_name + ' Sweep' + title_suffix)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return ax


def plot_multi_seed_distribution(ms_results, model_name='Default', save_path=None, show=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    seeds = ms_results['seeds']
    maes  = ms_results['maes']
    rmses = ms_results['rmses']
    ax1.bar(range(len(seeds)), maes, color='steelblue', alpha=0.7)
    ax1.axhline(y=ms_results['mae_mean'], color='r', linestyle='--', label='Mean=' + f"{ms_results['mae_mean']:.2f}")
    ax1.axhspan(ms_results['mae_mean'] - ms_results['mae_std'], ms_results['mae_mean'] + ms_results['mae_std'], alpha=0.15, color='red', label='+/- 1 std')
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('MAE (wins)')
    ax1.set_title(model_name + ': MAE across seeds')
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.legend(fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(range(len(seeds)), rmses, color='coral', alpha=0.7)
    ax2.axhline(y=ms_results['rmse_mean'], color='r', linestyle='--', label='Mean=' + f"{ms_results['rmse_mean']:.2f}")
    ax2.axhspan(ms_results['rmse_mean'] - ms_results['rmse_std'], ms_results['rmse_mean'] + ms_results['rmse_std'], alpha=0.15, color='red', label='+/- 1 std')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('RMSE (wins)')
    ax2.set_title(model_name + ': RMSE across seeds')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig

def plot_architecture_comparison(results_dict, save_path=None, show=False):
    names = list(results_dict.keys())
    maes = [results_dict[n]['mae']  for n in names]
    rmses = [results_dict[n]['rmse'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    ax.bar(x - width / 2, maes,  width, label='MAE',  alpha=0.8)
    ax.bar(x + width / 2, rmses, width, label='RMSE', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Error (wins)')
    ax.set_title('Architecture Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return ax

def plot_final_comparison(results_dict, save_path=None, show=False):
    labels = list(results_dict.keys())
    mae_means = [results_dict[l]['mae_mean']  for l in labels]
    mae_stds = [results_dict[l].get('mae_std', 0) for l in labels]
    rmse_means = [results_dict[l]['rmse_mean'] for l in labels]
    rmse_stds = [results_dict[l].get('rmse_std', 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    ax.bar(x - width / 2, mae_means,  width, yerr=mae_stds,  capsize=4, label='MAE',  alpha=0.8, color='steelblue')
    ax.bar(x + width / 2, rmse_means, width, yerr=rmse_stds, capsize=4, label='RMSE', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Error (wins)')
    ax.set_title('Final Model Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return ax


ARCH_CONFIGS = [
    ('1-layer [32]', 32, None),
    ('1-layer [64]', 64, None),
    ('1-layer [128]', 128, None),
    ('2-layer [64->32]', 64, 32),
    ('2-layer [128->64]', 128, 64),
    ('2-layer [64->16]', 64, 16)
]


def run_architecture_search(X_train_scaled, y_train, X_test_scaled, y_test, configs=None, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    if not configs:
        configs = ARCH_CONFIGS

    input_dim = X_train_scaled.shape[1]
    X_tr, y_tr, X_v, y_v = split_train_val(X_train_scaled, y_train)
    results = {}

    for name, h1, h2 in configs:
        print("  Training: " + name + " ...", end=" ", flush=True)
        set_seed(42)

        if not h2:
            model = WinPredictorOneLayer(input_dim, h1)
        else:
            model = WinPredictor(input_dim, hidden1=h1, hidden2=h2)

        history = train_model(model, X_tr, y_tr, X_v, y_v, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, patience=patience)
        preds = predict(history['model'], X_test_scaled)
        mae   = compute_mae(y_test, preds)
        rmse  = compute_rmse(y_test, preds)
        n_par = sum(p.numel() for p in history['model'].parameters())
        results[name] = {
            'mae': mae, 
            'rmse': rmse, 
            'n_params': n_par,
            'best_epoch': history['best_epoch']
        }
        print("MAE=" + f"{mae:.2f}" + " RMSE=" + f"{rmse:.2f}" + " params=" + str(n_par))

    return results


def run_nn(csv_path=None):
    data = prepare_data(csv_path)
    results = run_single(data['X_train_scaled'], data['y_train'], data['X_test_scaled'], data['y_test'], seed=42)

    plot_training_curve(results['train_losses'], results['val_losses'], results['best_epoch'], save_path=os.path.join(RESULTS_DIR, 'training_curve.png'))
    plot_pred_vs_actual(data['y_test'], results['preds'], model_name='Neural Network', save_path=os.path.join(RESULTS_DIR, 'nn_pred_vs_actual.png'))
    return results


if __name__ == '__main__':
    results = run_nn()
    print("Neural Network Results:")
    print("Architecture: 15 -> " + str(HIDDEN1) + " -> " + str(HIDDEN2) + " -> 1")
    print("Parameters: " + str(results['n_params']))
    print("Best epoch: " + str(results['best_epoch']) + " / " + str(results['epochs_run']))
    print("MAE: " + f"{results['mae']:.2f}" + " wins")
    print("RMSE: " + f"{results['rmse']:.2f}" + " wins")
