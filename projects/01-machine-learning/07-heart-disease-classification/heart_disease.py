"""
Heart Disease Classification
============================
Comprehensive classification project for predicting heart disease using
synthetic clinical data. Demonstrates multiple ML models, feature selection,
ensemble methods, and clinical threshold optimization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def generate_heart_disease_data(n_samples=1200):
    """Generate synthetic heart disease dataset with realistic clinical features."""
    age = np.random.normal(54, 9, n_samples).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n_samples)  # 68% male

    # Chest pain type (0-3): typical angina, atypical angina, non-anginal, asymptomatic
    chest_pain = np.random.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.15, 0.25, 0.45])

    resting_bp = np.random.normal(132, 18, n_samples).clip(90, 200).astype(int)
    cholesterol = np.random.normal(246, 52, n_samples).clip(125, 565).astype(int)
    fasting_bs = np.random.binomial(1, 0.15, n_samples)  # fasting blood sugar > 120

    # Resting ECG (0-2): normal, ST-T abnormality, LV hypertrophy
    resting_ecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15])

    max_hr = (220 - age - np.random.normal(0, 15, n_samples)).clip(70, 202).astype(int)
    exercise_angina = np.random.binomial(1, 0.33, n_samples)
    oldpeak = np.random.exponential(1.0, n_samples).clip(0, 6.2).round(1)

    # ST slope (0-2): upsloping, flat, downsloping
    st_slope = np.random.choice([0, 1, 2], n_samples, p=[0.35, 0.45, 0.20])

    # Generate target based on realistic clinical correlations
    risk_score = (
        0.03 * (age - 40)
        + 0.15 * sex
        + 0.25 * (chest_pain == 3).astype(float)
        + 0.10 * (chest_pain == 2).astype(float)
        + 0.01 * (resting_bp - 120) / 20
        + 0.008 * (cholesterol - 200) / 50
        + 0.15 * fasting_bs
        + 0.10 * (resting_ecg > 0).astype(float)
        - 0.01 * (max_hr - 120) / 20
        + 0.30 * exercise_angina
        + 0.15 * oldpeak
        + 0.20 * (st_slope == 1).astype(float)
        + 0.30 * (st_slope == 2).astype(float)
        + np.random.normal(0, 0.2, n_samples)
    )

    probability = 1 / (1 + np.exp(-2 * (risk_score - np.median(risk_score))))
    target = (np.random.random(n_samples) < probability).astype(int)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'chest_pain_type': chest_pain,
        'resting_bp': resting_bp, 'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_bs, 'resting_ecg': resting_ecg,
        'max_heart_rate': max_hr, 'exercise_angina': exercise_angina,
        'oldpeak': oldpeak, 'st_slope': st_slope, 'target': target
    })
    return df


def exploratory_data_analysis(df):
    """Perform comprehensive EDA on heart disease data."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    print(f"\nFeature statistics:\n{df.describe().round(2)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Age distribution by target
    for t in [0, 1]:
        subset = df[df['target'] == t]
        axes[0, 0].hist(subset['age'], alpha=0.6, label=f"{'Disease' if t else 'No Disease'}", bins=20)
    axes[0, 0].set_title('Age Distribution by Heart Disease')
    axes[0, 0].legend()

    # Chest pain type vs target
    ct = pd.crosstab(df['chest_pain_type'], df['target'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=axes[0, 1], colormap='RdYlGn_r')
    axes[0, 1].set_title('Chest Pain Type vs Heart Disease')
    axes[0, 1].set_xticklabels(['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'], rotation=45)

    # Max heart rate by target
    for t in [0, 1]:
        subset = df[df['target'] == t]
        axes[0, 2].hist(subset['max_heart_rate'], alpha=0.6,
                        label=f"{'Disease' if t else 'No Disease'}", bins=20)
    axes[0, 2].set_title('Max Heart Rate by Heart Disease')
    axes[0, 2].legend()

    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[1, 0], square=True, cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Feature Correlations')

    # Exercise angina vs target
    ct2 = pd.crosstab(df['exercise_angina'], df['target'], normalize='index')
    ct2.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='RdYlGn_r')
    axes[1, 1].set_title('Exercise Angina vs Heart Disease')
    axes[1, 1].set_xticklabels(['No', 'Yes'], rotation=0)

    # ST slope vs target
    ct3 = pd.crosstab(df['st_slope'], df['target'], normalize='index')
    ct3.plot(kind='bar', stacked=True, ax=axes[1, 2], colormap='RdYlGn_r')
    axes[1, 2].set_title('ST Slope vs Heart Disease')
    axes[1, 2].set_xticklabels(['Upsloping', 'Flat', 'Downsloping'], rotation=45)

    plt.tight_layout()
    plt.savefig('projects/01-machine-learning/07-heart-disease-classification/eda_plots.png', dpi=150)
    plt.close()
    print("\nEDA plots saved.")


def feature_selection_analysis(X, y, feature_names):
    """Perform feature selection using mutual information and RFE."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 60)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    print("\nMutual Information Scores:")
    print(mi_df.to_string(index=False))

    # Recursive Feature Elimination
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=6, step=1)
    rfe.fit(X, y)
    rfe_df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    print("\nRFE Rankings:")
    print(rfe_df.to_string(index=False))

    return mi_df, rfe_df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models and compare performance."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        use_scaled = name in ['Logistic Regression', 'KNN (k=5)', 'SVM (RBF)']
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='roc_auc')
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'model': model, 'accuracy': acc, 'f1': f1, 'auc': auc,
            'cv_auc_mean': cv_scores.mean(), 'cv_auc_std': cv_scores.std(),
            'y_pred': y_pred, 'y_prob': y_prob
        }

        print(f"\n{name}:")
        print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test Accuracy: {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")

    return results, scaler


def build_ensemble_models(X_train, X_test, y_train, y_test, scaler):
    """Build ensemble models: Voting and Stacking classifiers."""
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS")
    print("=" * 60)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Voting Classifier
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
    ]

    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_vote = voting_clf.predict(X_test_scaled)
    y_prob_vote = voting_clf.predict_proba(X_test_scaled)[:, 1]

    vote_acc = accuracy_score(y_test, y_pred_vote)
    vote_f1 = f1_score(y_test, y_pred_vote)
    vote_auc = roc_auc_score(y_test, y_prob_vote)

    print(f"\nVoting Classifier (Soft):")
    print(f"  Accuracy: {vote_acc:.4f}  |  F1: {vote_f1:.4f}  |  AUC: {vote_auc:.4f}")

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )
    stacking_clf.fit(X_train_scaled, y_train)
    y_pred_stack = stacking_clf.predict(X_test_scaled)
    y_prob_stack = stacking_clf.predict_proba(X_test_scaled)[:, 1]

    stack_acc = accuracy_score(y_test, y_pred_stack)
    stack_f1 = f1_score(y_test, y_pred_stack)
    stack_auc = roc_auc_score(y_test, y_prob_stack)

    print(f"\nStacking Classifier:")
    print(f"  Accuracy: {stack_acc:.4f}  |  F1: {stack_f1:.4f}  |  AUC: {stack_auc:.4f}")

    return {
        'Voting': {'y_prob': y_prob_vote, 'auc': vote_auc, 'f1': vote_f1},
        'Stacking': {'y_prob': y_prob_stack, 'auc': stack_auc, 'f1': stack_f1}
    }


def clinical_threshold_optimization(y_test, y_prob, model_name='Best Model'):
    """Optimize classification threshold for clinical use (sensitivity vs specificity)."""
    print("\n" + "=" * 60)
    print("CLINICAL THRESHOLD OPTIMIZATION")
    print("=" * 60)

    thresholds = np.arange(0.1, 0.91, 0.05)
    metrics = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_test == 1))
        tn = np.sum((y_pred_t == 0) & (y_test == 0))
        fp = np.sum((y_pred_t == 1) & (y_test == 0))
        fn = np.sum((y_pred_t == 0) & (y_test == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics.append({
            'threshold': t, 'sensitivity': sensitivity,
            'specificity': specificity, 'ppv': ppv, 'npv': npv
        })

    metrics_df = pd.DataFrame(metrics)

    # Find optimal threshold (maximize Youden's J statistic)
    metrics_df['youden_j'] = metrics_df['sensitivity'] + metrics_df['specificity'] - 1
    optimal_idx = metrics_df['youden_j'].idxmax()
    optimal = metrics_df.loc[optimal_idx]

    print(f"\nOptimal threshold (Youden's J): {optimal['threshold']:.2f}")
    print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
    print(f"  Specificity: {optimal['specificity']:.4f}")
    print(f"  PPV: {optimal['ppv']:.4f}")
    print(f"  NPV: {optimal['npv']:.4f}")

    # High-sensitivity threshold (for screening - minimize missed cases)
    high_sens = metrics_df[metrics_df['sensitivity'] >= 0.90].iloc[-1] if len(metrics_df[metrics_df['sensitivity'] >= 0.90]) > 0 else metrics_df.iloc[0]
    print(f"\nHigh-sensitivity threshold (>=90% sensitivity): {high_sens['threshold']:.2f}")
    print(f"  Sensitivity: {high_sens['sensitivity']:.4f}, Specificity: {high_sens['specificity']:.4f}")

    # Plot sensitivity vs specificity tradeoff
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(metrics_df['threshold'], metrics_df['sensitivity'], 'b-', label='Sensitivity', linewidth=2)
    ax.plot(metrics_df['threshold'], metrics_df['specificity'], 'r-', label='Specificity', linewidth=2)
    ax.plot(metrics_df['threshold'], metrics_df['ppv'], 'g--', label='PPV', linewidth=1.5)
    ax.plot(metrics_df['threshold'], metrics_df['npv'], 'm--', label='NPV', linewidth=1.5)
    ax.axvline(x=optimal['threshold'], color='k', linestyle=':', alpha=0.5, label=f"Optimal ({optimal['threshold']:.2f})")
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Clinical Threshold Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('projects/01-machine-learning/07-heart-disease-classification/threshold_optimization.png', dpi=150)
    plt.close()

    return metrics_df


def plot_model_comparison(results, ensemble_results, y_test):
    """Plot ROC curves and model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC curves
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

    for name, res in ensemble_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, '--', label=f"{name} (AUC={res['auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves - All Models')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Bar chart comparison
    all_models = {**{k: v for k, v in results.items()}, **ensemble_results}
    names = list(all_models.keys())
    aucs = [all_models[n]['auc'] for n in names]
    f1s = [all_models[n]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, aucs, width, label='AUC', color='steelblue')
    axes[1].bar(x + width/2, f1s, width, label='F1', color='coral')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Model Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig('projects/01-machine-learning/07-heart-disease-classification/model_comparison.png', dpi=150)
    plt.close()
    print("\nModel comparison plots saved.")


def main():
    """Main execution pipeline."""
    print("Heart Disease Classification Pipeline")
    print("=" * 60)

    # Generate data
    df = generate_heart_disease_data(1200)

    # EDA
    exploratory_data_analysis(df)

    # Prepare features
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Feature selection
    feature_selection_analysis(X.values, y.values, feature_names)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, stratify=y, random_state=42
    )

    # Train models
    results, scaler = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Ensemble methods
    ensemble_results = build_ensemble_models(X_train, X_test, y_train, y_test, scaler)

    # Find best model
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\nBest individual model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    # Clinical threshold optimization using best model's probabilities
    clinical_threshold_optimization(y_test, results[best_name]['y_prob'], best_name)

    # Comparison plots
    plot_model_comparison(results, ensemble_results, y_test)

    # Best model confusion matrix
    best_pred = results[best_name]['y_pred']
    print(f"\nBest Model Confusion Matrix ({best_name}):")
    cm = confusion_matrix(y_test, best_pred)
    print(cm)
    print(f"\n{classification_report(y_test, best_pred, target_names=['No Disease', 'Heart Disease'])}")

    print("\n" + "=" * 60)
    print("Pipeline complete. All results and visualizations saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
