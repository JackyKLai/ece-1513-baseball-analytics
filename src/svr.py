"""
Support Vector Regression (SVR) model for MLB win prediction.

Based on Lectures 8–10 (SVC/SVM, kernel methods).

TODO:
    - Implement fit_svr() with grid search over kernel, C, epsilon
    - Compare linear vs RBF kernels
    - Discuss why nonlinear kernels might capture feature interactions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from data_pipeline import prepare_data, _REPO_ROOT
from evaluate import compute_mae, compute_rmse


def fit_svr(X_train, y_train, cv=5):
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1.0, 10.0, 50.0],
        'epsilon': [0.1, 0.5, 1.0, 2.0],
        'gamma': ['scale', 0.01, 0.1],
    }

    grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def _best_row_for_kernel(cv_results_df, kernel_name):
    rows = cv_results_df[cv_results_df['param_kernel'] == kernel_name]
    if rows.empty:
        return None
    row = rows.loc[rows['rank_test_score'].idxmin()]
    return {
        'kernel': kernel_name,
        'mae_cv': float(-row['mean_test_score']),
        'params': {
            'C': float(row['param_C']),
            'epsilon': float(row['param_epsilon']),
            'gamma': str(row['param_gamma']),
        },
    }


def _plot_svr_diagnostics(y_true, y_pred, save_path=None, show=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_errors = np.abs(y_pred - y_true)

    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))

    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    scatter = ax.scatter(
        y_true,
        y_pred,
        c=abs_errors,
        cmap='viridis',
        alpha=0.90,
        edgecolors='k',
        linewidths=0.45,
        s=42,
    )
    ax.plot([lo, hi], [lo, hi], color='#C44E52', linestyle='--', linewidth=1.4, label='Perfect prediction')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Actual Wins')
    ax.set_ylabel('Predicted Wins')
    ax.set_title('SVR Prediction vs Actual', fontsize=13, pad=10)
    ax.grid(alpha=0.22)
    ax.legend(loc='upper left', frameon=True, framealpha=0.92)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error')

    fig.suptitle(
        f"SVR Diagnostics | MAE={mae:.3f}, RMSE={rmse:.3f}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_svr_residual_hist(y_true, y_pred, save_path=None, show=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(residuals, bins=16, color='#4C72B0', alpha=0.85, edgecolor='black')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1.2, label='Zero error')
    ax.axvline(np.mean(residuals), color='black', linestyle=':', linewidth=1.2, label='Mean residual')
    ax.set_xlabel('Residual (Predicted - Actual)')
    ax.set_ylabel('Count')
    ax.set_title('SVR Residual Distribution')
    ax.grid(alpha=0.2, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_kernel_cv_mae(kernel_summary, save_path=None, show=False):
    kernel_names = []
    mae_values = []
    for kernel_name in ['linear', 'rbf']:
        row = kernel_summary.get(kernel_name)
        if row is None:
            continue
        kernel_names.append(kernel_name.upper())
        mae_values.append(float(row['mae_cv']))

    if not kernel_names:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(kernel_names, mae_values, color=['#4C72B0', '#55A868'], alpha=0.9, edgecolor='black')
    ax.set_ylabel('CV MAE (lower is better)')
    ax.set_title('Linear vs RBF (Best CV MAE)')
    ax.grid(alpha=0.2, axis='y')
    ax.set_ylim(0.0, max(mae_values) * 1.18)
    for bar, value in zip(bars, mae_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(mae_values) * 0.03,
            f'{value:.3f}',
            ha='center',
            va='bottom',
        )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_error_table(y_true, y_pred):
    df = pd.DataFrame({
        'actual_wins': np.asarray(y_true, dtype=float),
        'predicted_wins': np.asarray(y_pred, dtype=float),
    })
    df['error'] = df['predicted_wins'] - df['actual_wins']
    df['abs_error'] = df['error'].abs()
    return df.sort_values('abs_error', ascending=False).reset_index(drop=True)


def print_svr_report(results, top_k_errors=8):
    print("=== SVR Experiment Report ===")
    print(f"Best params: {results['best_params']}")
    print(f"Test MAE : {results['mae']:.3f}")
    print(f"Test RMSE: {results['rmse']:.3f}")
    print()
    print("Kernel-wise best CV MAE:")
    kernel_rows = []
    for kernel_name in ['linear', 'rbf']:
        row = results['kernel_summary'][kernel_name]
        if row is None:
            continue
        kernel_rows.append({
            'kernel': kernel_name,
            'mae_cv': round(row['mae_cv'], 3),
            'C': row['params']['C'],
            'epsilon': row['params']['epsilon'],
            'gamma': row['params']['gamma'],
        })
    if kernel_rows:
        print(pd.DataFrame(kernel_rows).to_string(index=False))
    print()
    print(f"Top {top_k_errors} largest absolute errors (test set):")
    print(results['error_table'].head(top_k_errors).to_string(index=False))


def run_svr_experiment(csv_path=None, cv=5, save_plot=True, show_plot=False):
    data = prepare_data(csv_path)
    X_train = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data['X_test_scaled']
    y_test = data['y_test']

    grid = fit_svr(X_train, y_train, cv=cv)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    mae = compute_mae(y_test, y_pred)
    rmse = compute_rmse(y_test, y_pred)
    error_table = _build_error_table(y_test, y_pred)

    save_path = None
    residual_save_path = None
    kernel_save_path = None
    if save_plot:
        save_path = os.path.join(_REPO_ROOT, 'results', 'svr', 'svr_pred_vs_actual.png')
        residual_save_path = os.path.join(_REPO_ROOT, 'results', 'svr', 'svr_residual_hist.png')
        kernel_save_path = os.path.join(_REPO_ROOT, 'results', 'svr', 'svr_kernel_cv_mae.png')
    _plot_svr_diagnostics(
        y_test,
        y_pred,
        save_path=save_path,
        show=show_plot,
    )
    _plot_svr_residual_hist(
        y_test,
        y_pred,
        save_path=residual_save_path,
        show=show_plot,
    )

    cv_df = pd.DataFrame(grid.cv_results_)
    linear_best = _best_row_for_kernel(cv_df, 'linear')
    rbf_best = _best_row_for_kernel(cv_df, 'rbf')
    kernel_summary = {
        'linear': linear_best,
        'rbf': rbf_best,
    }
    _plot_kernel_cv_mae(
        kernel_summary=kernel_summary,
        save_path=kernel_save_path,
        show=show_plot,
    )

    return {
        'model': model,
        'best_params': grid.best_params_,
        'kernel_summary': kernel_summary,
        'mae': mae,
        'rmse': rmse,
        'y_test': y_test,
        'y_pred': y_pred,
        'error_table': error_table,
    }


if __name__ == '__main__':
    results = run_svr_experiment(save_plot=True, show_plot=False)
    print_svr_report(results)
