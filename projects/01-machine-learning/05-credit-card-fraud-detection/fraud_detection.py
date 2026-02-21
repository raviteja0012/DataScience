"""
Project 5: Credit Card Fraud Detection
========================================
Synthetic credit card transaction data with extreme class imbalance (99.8% normal, 0.2% fraud).
Demonstrates SMOTE oversampling, class weighting, threshold tuning, and precision-recall evaluation.

Models: Logistic Regression, Random Forest, Isolation Forest, XGBoost/GradientBoosting
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, make_scorer
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imbalanced-learn not installed. SMOTE will be skipped.")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("INFO: xgboost not installed. Using sklearn GradientBoostingClassifier instead.")


# =============================================================================
# 1. Synthetic Data Generation
# =============================================================================
def generate_fraud_data(n_samples=10000, fraud_ratio=0.002, random_state=42):
    """
    Generate synthetic credit card transaction data.
    99.8% normal transactions, 0.2% fraudulent.
    Features mimic PCA-transformed components (V1-V10) plus Amount and Time.
    """
    np.random.seed(random_state)

    n_fraud = max(int(n_samples * fraud_ratio), 5)
    n_normal = n_samples - n_fraud

    print(f"Generating {n_samples} transactions: {n_normal} normal, {n_fraud} fraud")
    print(f"Fraud ratio: {fraud_ratio * 100:.1f}%\n")

    # --- Normal transactions ---
    normal_features = np.random.randn(n_normal, 10) * 1.0
    normal_amount = np.abs(np.random.lognormal(mean=3.5, sigma=1.2, size=n_normal))
    normal_time = np.sort(np.random.uniform(0, 172800, size=n_normal))  # 2 days in seconds

    # --- Fraudulent transactions ---
    # Fraudulent transactions have shifted distributions
    fraud_features = np.random.randn(n_fraud, 10) * 1.5
    fraud_features[:, 0] -= 3.0   # V1 shifts negative
    fraud_features[:, 1] += 2.5   # V2 shifts positive
    fraud_features[:, 2] -= 2.0   # V3 shifts negative
    fraud_features[:, 4] += 2.0   # V5 shifts positive
    fraud_features[:, 6] -= 1.5   # V7 shifts negative

    fraud_amount = np.abs(np.random.lognormal(mean=5.0, sigma=1.5, size=n_fraud))
    fraud_time = np.random.uniform(0, 172800, size=n_fraud)

    # Combine
    v_cols = [f'V{i}' for i in range(1, 11)]
    normal_df = pd.DataFrame(normal_features, columns=v_cols)
    normal_df['Amount'] = normal_amount
    normal_df['Time'] = normal_time
    normal_df['Class'] = 0

    fraud_df = pd.DataFrame(fraud_features, columns=v_cols)
    fraud_df['Amount'] = fraud_amount
    fraud_df['Time'] = fraud_time
    fraud_df['Class'] = 1

    df = pd.concat([normal_df, fraud_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================
def exploratory_analysis(df):
    """Perform EDA on the fraud dataset."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    class_counts = df['Class'].value_counts()
    for cls, count in class_counts.items():
        label = "Fraud" if cls == 1 else "Normal"
        print(f"  {label} (Class {cls}): {count} ({count/len(df)*100:.3f}%)")

    print(f"\nFeature statistics:")
    print(df.describe().round(3).to_string())

    # Plot 1: Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Class distribution bar plot
    ax = axes[0, 0]
    colors = ['#2ecc71', '#e74c3c']
    class_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Class (0=Normal, 1=Fraud)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    for i, (idx, val) in enumerate(class_counts.items()):
        ax.text(i, val * 1.1, str(val), ha='center', fontweight='bold')

    # Amount distribution by class
    ax = axes[0, 1]
    for cls, color, label in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['Class'] == cls]['Amount']
        ax.hist(subset, bins=50, alpha=0.7, color=color, label=label, density=True)
    ax.set_title('Transaction Amount Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Amount ($)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(0, df['Amount'].quantile(0.99))

    # Time distribution by class
    ax = axes[1, 0]
    for cls, color, label in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['Class'] == cls]['Time']
        ax.hist(subset, bins=50, alpha=0.7, color=color, label=label, density=True)
    ax.set_title('Transaction Time Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Density')
    ax.legend()

    # Correlation heatmap for top features
    ax = axes[1, 1]
    corr_with_class = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
    top_features = corr_with_class.head(8).index.tolist()
    corr_matrix = df[top_features + ['Class']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation: Top Features vs Class', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/DataScience/projects/01-machine-learning/05-credit-card-fraud-detection/eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nEDA plots saved to eda_plots.png")


# =============================================================================
# 3. Data Preprocessing
# =============================================================================
def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data: scale features, handle imbalance, train/test split."""
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].copy()
    y = df['Class'].copy()

    # Use RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"  Normal: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()}")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"  Normal: {(y_test == 0).sum()}, Fraud: {(y_test == 1).sum()}")

    # Apply SMOTE to training data only
    if HAS_IMBLEARN:
        smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"\nAfter SMOTE resampling:")
        print(f"  Training set: {X_train_resampled.shape[0]} samples")
        print(f"  Normal: {(y_train_resampled == 0).sum()}, Fraud: {(y_train_resampled == 1).sum()}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print("\nSMOTE skipped (imbalanced-learn not available).")

    return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, feature_cols


# =============================================================================
# 4. Model Training & Evaluation
# =============================================================================
def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                               X_train_resampled, y_train_resampled, feature_cols):
    """Train multiple models and evaluate on precision-recall and ROC metrics."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    results = {}

    # --- Model 1: Logistic Regression (with class_weight) ---
    print("\n--- Logistic Regression (class_weight='balanced') ---")
    lr = LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1, random_state=42, solver='lbfgs'
    )
    lr.fit(X_train_resampled, y_train_resampled)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_lr, target_names=['Normal', 'Fraud']))
    results['Logistic Regression'] = {
        'model': lr, 'y_pred': y_pred_lr, 'y_prob': y_prob_lr,
        'roc_auc': roc_auc_score(y_test, y_prob_lr),
        'avg_precision': average_precision_score(y_test, y_prob_lr),
        'f1': f1_score(y_test, y_pred_lr),
    }

    # --- Model 2: Random Forest (with class_weight) ---
    print("\n--- Random Forest (class_weight='balanced_subsample') ---")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced_subsample',
        max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Fraud']))
    results['Random Forest'] = {
        'model': rf, 'y_pred': y_pred_rf, 'y_prob': y_prob_rf,
        'roc_auc': roc_auc_score(y_test, y_prob_rf),
        'avg_precision': average_precision_score(y_test, y_prob_rf),
        'f1': f1_score(y_test, y_pred_rf),
    }

    # --- Model 3: Isolation Forest (unsupervised anomaly detection) ---
    print("\n--- Isolation Forest (unsupervised) ---")
    iso = IsolationForest(
        n_estimators=200, contamination=0.005,
        max_samples='auto', random_state=42, n_jobs=-1
    )
    iso.fit(X_train)  # Train on original (unbalanced) data

    # Isolation Forest returns -1 for anomalies, 1 for normal
    iso_pred_raw = iso.predict(X_test)
    y_pred_iso = np.where(iso_pred_raw == -1, 1, 0)  # Convert to 0/1
    iso_scores = -iso.score_samples(X_test)  # Higher = more anomalous

    print(classification_report(y_test, y_pred_iso, target_names=['Normal', 'Fraud']))
    results['Isolation Forest'] = {
        'model': iso, 'y_pred': y_pred_iso, 'y_prob': iso_scores,
        'roc_auc': roc_auc_score(y_test, iso_scores),
        'avg_precision': average_precision_score(y_test, iso_scores),
        'f1': f1_score(y_test, y_pred_iso),
    }

    # --- Model 4: XGBoost / Gradient Boosting ---
    if HAS_XGBOOST:
        print("\n--- XGBoost ---")
        n_neg = (y_train_resampled == 0).sum()
        n_pos = (y_train_resampled == 1).sum()
        scale_pos = n_neg / max(n_pos, 1)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos, eval_metric='aucpr',
            random_state=42, use_label_encoder=False
        )
        model_name = 'XGBoost'
    else:
        print("\n--- Gradient Boosting ---")
        xgb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42
        )
        model_name = 'Gradient Boosting'

    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Fraud']))
    results[model_name] = {
        'model': xgb, 'y_pred': y_pred_xgb, 'y_prob': y_prob_xgb,
        'roc_auc': roc_auc_score(y_test, y_prob_xgb),
        'avg_precision': average_precision_score(y_test, y_prob_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
    }

    return results


# =============================================================================
# 5. Confusion Matrices
# =============================================================================
def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models."""
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        ax.set_title(f'{name}\nF1={res["f1"]:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name}:")
        print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
        print(f"  False Negative Rate: {fn/(fn+tp):.4f}" if (fn+tp) > 0 else "  No positive samples")

    plt.tight_layout()
    plt.savefig('/home/user/DataScience/projects/01-machine-learning/05-credit-card-fraud-detection/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrices saved to confusion_matrices.png")


# =============================================================================
# 6. ROC & Precision-Recall Curves
# =============================================================================
def plot_roc_and_pr_curves(results, y_test):
    """Plot ROC and Precision-Recall curves."""
    print("\n" + "=" * 70)
    print("ROC & PRECISION-RECALL CURVES")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    # ROC Curve
    ax = axes[0]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax = axes[1]
    for (name, res), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AP={res['avg_precision']:.3f})")

    # Baseline (prevalence)
    prevalence = y_test.sum() / len(y_test)
    ax.axhline(y=prevalence, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({prevalence:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/DataScience/projects/01-machine-learning/05-credit-card-fraud-detection/roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("ROC and Precision-Recall curves saved to roc_pr_curves.png")

    # Print summary
    print("\n--- Model Comparison Summary ---")
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'ROC AUC': res['roc_auc'],
            'Avg Precision': res['avg_precision'],
            'F1 Score': res['f1'],
        })
    summary_df = pd.DataFrame(summary_data).sort_values('Avg Precision', ascending=False)
    print(summary_df.to_string(index=False))


# =============================================================================
# 7. Threshold Tuning
# =============================================================================
def threshold_tuning(results, y_test):
    """Find optimal classification thresholds for each model."""
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING")
    print("=" * 70)

    # Focus on the best supervised model
    supervised_models = {k: v for k, v in results.items() if k != 'Isolation Forest'}
    best_model_name = max(supervised_models, key=lambda k: supervised_models[k]['avg_precision'])
    best_res = supervised_models[best_model_name]

    print(f"\nTuning threshold for best model: {best_model_name}")

    # Compute precision-recall at various thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_res['y_prob'])

    # Find threshold that maximizes F1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"  Default threshold (0.5): F1={f1_score(y_test, best_res['y_pred']):.4f}")
    print(f"  Optimal threshold: {best_threshold:.4f}")
    print(f"  Optimal F1 score: {best_f1:.4f}")
    print(f"  Precision at optimal: {precisions[best_idx]:.4f}")
    print(f"  Recall at optimal: {recalls[best_idx]:.4f}")

    # Plot threshold analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 vs Threshold
    ax = axes[0]
    ax.plot(thresholds, f1_scores, 'b-', lw=2)
    ax.axvline(x=best_threshold, color='r', linestyle='--', lw=1.5,
               label=f'Optimal={best_threshold:.3f}')
    ax.axvline(x=0.5, color='gray', linestyle='--', lw=1, label='Default=0.5')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'F1 Score vs Threshold ({best_model_name})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision-Recall vs Threshold
    ax = axes[1]
    ax.plot(thresholds, precisions[:-1], 'b-', lw=2, label='Precision')
    ax.plot(thresholds, recalls[:-1], 'r-', lw=2, label='Recall')
    ax.axvline(x=best_threshold, color='green', linestyle='--', lw=1.5,
               label=f'Optimal={best_threshold:.3f}')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision & Recall vs Threshold ({best_model_name})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/DataScience/projects/01-machine-learning/05-credit-card-fraud-detection/threshold_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Threshold tuning plots saved to threshold_tuning.png")

    # Apply optimal threshold and show updated confusion matrix
    y_pred_optimal = (best_res['y_prob'] >= best_threshold).astype(int)
    print(f"\nClassification report at optimal threshold ({best_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal, target_names=['Normal', 'Fraud']))

    return best_threshold


# =============================================================================
# 8. Feature Importance
# =============================================================================
def feature_importance_analysis(results, feature_cols):
    """Analyze feature importance from tree-based models."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, model_name in enumerate(['Random Forest']):
        if model_name in results:
            model = results[model_name]['model']
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            ax = axes[0]
            ax.barh(range(len(importances)), importances[indices], color='#3498db')
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels([feature_cols[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} Feature Importance', fontsize=13, fontweight='bold')
            ax.invert_yaxis()

            print(f"\n{model_name} top features:")
            for i in range(min(5, len(importances))):
                print(f"  {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

    # Logistic Regression coefficients
    if 'Logistic Regression' in results:
        lr_model = results['Logistic Regression']['model']
        coefs = np.abs(lr_model.coef_[0])
        indices = np.argsort(coefs)[::-1]

        ax = axes[1]
        colors = ['#e74c3c' if lr_model.coef_[0][i] < 0 else '#2ecc71' for i in indices]
        ax.barh(range(len(coefs)), lr_model.coef_[0][indices], color=colors)
        ax.set_yticks(range(len(coefs)))
        ax.set_yticklabels([feature_cols[i] for i in indices])
        ax.set_xlabel('Coefficient')
        ax.set_title('Logistic Regression Coefficients', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linestyle='-', lw=0.5)

        print(f"\nLogistic Regression top features (by |coefficient|):")
        for i in range(min(5, len(coefs))):
            print(f"  {feature_cols[indices[i]]}: {lr_model.coef_[0][indices[i]]:.4f}")

    plt.tight_layout()
    plt.savefig('/home/user/DataScience/projects/01-machine-learning/05-credit-card-fraud-detection/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFeature importance plots saved to feature_importance.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CREDIT CARD FRAUD DETECTION")
    print("  Handling Extreme Class Imbalance with Multiple Models")
    print("=" * 70)

    # 1. Generate data
    df = generate_fraud_data(n_samples=10000, fraud_ratio=0.002, random_state=42)

    # 2. EDA
    exploratory_analysis(df)

    # 3. Preprocess
    X_train, X_test, y_train, y_test, X_train_res, y_train_res, feature_cols = preprocess_data(df)

    # 4. Train and evaluate models
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X_train_res, y_train_res, feature_cols
    )

    # 5. Confusion matrices
    plot_confusion_matrices(results, y_test)

    # 6. ROC and PR curves
    plot_roc_and_pr_curves(results, y_test)

    # 7. Threshold tuning
    best_threshold = threshold_tuning(results, y_test)

    # 8. Feature importance
    feature_importance_analysis(results, feature_cols)

    print("\n" + "=" * 70)
    print("  FRAUD DETECTION ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  - Precision-Recall AUC is more informative than ROC AUC for imbalanced data")
    print("  - SMOTE + class_weight helps models learn the minority class")
    print("  - Threshold tuning can significantly improve detection at acceptable FPR")
    print("  - Isolation Forest provides unsupervised anomaly detection baseline")
    print(f"  - Optimal threshold found: {best_threshold:.4f}")


if __name__ == '__main__':
    main()
