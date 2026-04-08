import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

from data_pipeline import prepare_data, _REPO_ROOT
from evaluate import compute_mae, compute_rmse, plot_pred_vs_actual

KERNELS_TO_COMPARE = ['linear', 'rbf', 'poly']


def fit_svr(X_train, y_train, cv=5, shuffle_cv=False, cv_random_state=None, n_jobs=-1):
    cv_strategy = cv
    if isinstance(cv, int) and shuffle_cv:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=cv_random_state)

    base_grid = {
        'C': [0.1, 1.0, 10.0, 50.0],
        'epsilon': [0.1, 0.5, 1.0, 2.0],
    }
    param_grid = []
    if 'linear' in KERNELS_TO_COMPARE:
        param_grid.append({
            'kernel': ['linear'],
            **base_grid,
        })
    if 'rbf' in KERNELS_TO_COMPARE:
        param_grid.append({
            'kernel': ['rbf'],
            **base_grid,
            'gamma': ['scale', 0.01, 0.1],
        })
    if 'poly' in KERNELS_TO_COMPARE:
        param_grid.append({
            'kernel': ['poly'],
            **base_grid,
            'gamma': ['scale', 0.01, 0.1],
            'degree': [2, 3],
            'coef0': [0.0, 1.0],
        })

    grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=cv_strategy,
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def _best_row_for_kernel(cv_results_df, kernel_name):
    rows = cv_results_df[cv_results_df['param_kernel'] == kernel_name]
    if rows.empty:
        return None
    row = rows.loc[rows['rank_test_score'].idxmin()]
    gamma = row['param_gamma'] if 'param_gamma' in row.index else None
    degree = row['param_degree'] if 'param_degree' in row.index else None
    coef0 = row['param_coef0'] if 'param_coef0' in row.index else None
    if pd.isna(gamma):
        gamma = None
    if pd.isna(degree):
        degree = None
    if pd.isna(coef0):
        coef0 = None
    return {
        'kernel': kernel_name,
        'mae_cv': float(-row['mean_test_score']),
        'params': {
            'C': float(row['param_C']),
            'epsilon': float(row['param_epsilon']),
            'gamma': gamma if gamma is None else str(gamma),
            'degree': None if degree is None else int(degree),
            'coef0': None if coef0 is None else float(coef0),
        },
    }


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
    for kernel_name in KERNELS_TO_COMPARE:
        row = kernel_summary.get(kernel_name)
        if row is None:
            continue
        kernel_names.append(kernel_name.upper())
        mae_values.append(float(row['mae_cv']))

    if not kernel_names:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    color_map = {
        'LINEAR': '#4C72B0',
        'RBF': '#55A868',
        'POLY': '#C44E52',
    }
    bar_colors = [color_map.get(name, '#8172B2') for name in kernel_names]
    bars = ax.bar(kernel_names, mae_values, color=bar_colors, alpha=0.9, edgecolor='black')
    ax.set_ylabel('CV MAE (lower is better)')
    ax.set_title('Kernel Comparison (Best CV MAE)')
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
    for kernel_name in KERNELS_TO_COMPARE:
        row = results['kernel_summary'][kernel_name]
        if row is None:
            continue
        kernel_rows.append({
            'kernel': kernel_name,
            'mae_cv': round(row['mae_cv'], 3),
            'C': row['params']['C'],
            'epsilon': row['params']['epsilon'],
            'gamma': row['params']['gamma'],
            'degree': row['params']['degree'],
            'coef0': row['params']['coef0'],
        })
    if kernel_rows:
        print(pd.DataFrame(kernel_rows).to_string(index=False))
    print()
    print(f"Top {top_k_errors} largest absolute errors (test set):")
    print(results['error_table'].head(top_k_errors).to_string(index=False))


def run_svr_experiment(
    csv_path=None,
    cv=5,
    save_plot=True,
    show_plot=False,
    shuffle_cv=False,
    cv_random_state=None,
    n_jobs=-1,
):
    data = prepare_data(csv_path)
    X_train = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data['X_test_scaled']
    y_test = data['y_test']

    grid = fit_svr(
        X_train,
        y_train,
        cv=cv,
        shuffle_cv=shuffle_cv,
        cv_random_state=cv_random_state,
        n_jobs=n_jobs,
    )
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
    plot_pred_vs_actual(
        y_test,
        y_pred,
        model_name='SVR',
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
    poly_best = _best_row_for_kernel(cv_df, 'poly')
    kernel_summary = {
        'linear': linear_best,
        'rbf': rbf_best,
        'poly': poly_best,
    }
    _plot_kernel_cv_mae(
        kernel_summary=kernel_summary,
        save_path=kernel_save_path,
        show=show_plot,
    )

    return {
        'model': model,
        'best_params': grid.best_params_,
        'best_cv_mae': float(-grid.best_score_),
        'kernel_summary': kernel_summary,
        'mae': mae,
        'rmse': rmse,
        'y_test': y_test,
        'y_pred': y_pred,
        'error_table': error_table,
    }


def run_svr_multi_seed(csv_path=None, seeds=range(10), cv=5, n_jobs=-1):
    rows = []
    for seed in seeds:
        result = run_svr_experiment(
            csv_path=csv_path,
            cv=cv,
            save_plot=False,
            show_plot=False,
            shuffle_cv=True,
            cv_random_state=seed,
            n_jobs=n_jobs,
        )
        rows.append({
            'seed': int(seed),
            'kernel': result['best_params']['kernel'],
            'C': float(result['best_params']['C']),
            'epsilon': float(result['best_params']['epsilon']),
            'gamma': result['best_params'].get('gamma'),
            'degree': result['best_params'].get('degree'),
            'coef0': result['best_params'].get('coef0'),
            'cv_mae': float(result['best_cv_mae']),
            'test_mae': float(result['mae']),
            'test_rmse': float(result['rmse']),
        })

    df = pd.DataFrame(rows)
    summary = {
        'n_runs': int(len(df)),
        'test_mae_mean': float(df['test_mae'].mean()),
        'test_mae_std': float(df['test_mae'].std(ddof=0)),
        'test_rmse_mean': float(df['test_rmse'].mean()),
        'test_rmse_std': float(df['test_rmse'].std(ddof=0)),
        'cv_mae_mean': float(df['cv_mae'].mean()),
        'cv_mae_std': float(df['cv_mae'].std(ddof=0)),
    }
    return {
        'per_seed': df.sort_values('seed').reset_index(drop=True),
        'summary': summary,
    }


def print_svr_multi_seed_report(multi_seed_results):
    summary = multi_seed_results['summary']
    print("=== SVR Multi-Seed Report ===")
    print(f"Runs: {summary['n_runs']}")
    print(f"CV MAE  : {summary['cv_mae_mean']:.3f} ± {summary['cv_mae_std']:.3f}")
    print(f"Test MAE: {summary['test_mae_mean']:.3f} ± {summary['test_mae_std']:.3f}")
    print(f"Test RMSE: {summary['test_rmse_mean']:.3f} ± {summary['test_rmse_std']:.3f}")
    print()
    print("Per-seed best params and metrics:")
    print(multi_seed_results['per_seed'].to_string(index=False))


if __name__ == '__main__':
    results = run_svr_experiment(save_plot=True, show_plot=False)
    print_svr_report(results)
