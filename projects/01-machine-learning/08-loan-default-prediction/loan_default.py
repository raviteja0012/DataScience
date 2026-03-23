"""
Loan Default Prediction
=======================
Risk scoring and loan default prediction using synthetic financial data.
Demonstrates probability calibration, scorecard development, profit curve
analysis, and model fairness assessment.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = 'projects/01-machine-learning/08-loan-default-prediction'


def generate_loan_data(n_samples=5000):
    """Generate synthetic loan application data."""
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).clip(20000, 500000).astype(int)
    credit_score = np.random.normal(680, 80, n_samples).clip(300, 850).astype(int)
    loan_amount = (annual_income * np.random.uniform(0.1, 0.8, n_samples)).clip(1000, 500000).astype(int)
    employment_length = np.random.exponential(5, n_samples).clip(0, 40).astype(int)
    debt_to_income = np.random.beta(2, 5, n_samples) * 60
    num_credit_lines = np.random.poisson(5, n_samples).clip(0, 25)
    delinquencies_2yr = np.random.poisson(0.3, n_samples).clip(0, 10)
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.35, 0.15, 0.50])
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'other'],
        n_samples, p=[0.40, 0.20, 0.15, 0.10, 0.15]
    )
    interest_rate = (20 - 0.015 * credit_score + 0.05 * debt_to_income
                     + np.random.normal(0, 1.5, n_samples)).clip(3, 30)
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45])

    # Default probability based on risk factors
    risk_score = (
        -0.003 * credit_score
        + 0.02 * debt_to_income
        - 0.03 * employment_length
        + 0.15 * delinquencies_2yr
        + 0.00001 * (loan_amount / annual_income * 100)
        + 0.05 * interest_rate
        + np.random.normal(0, 0.3, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-(risk_score + 0.5)))
    default = (np.random.random(n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        'annual_income': annual_income, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'employment_length': employment_length,
        'debt_to_income': debt_to_income.round(2),
        'num_credit_lines': num_credit_lines,
        'delinquencies_2yr': delinquencies_2yr,
        'home_ownership': home_ownership, 'loan_purpose': loan_purpose,
        'interest_rate': interest_rate.round(2), 'gender': gender,
        'default': default
    })
    return df


def calculate_woe_iv(df, feature, target, bins=10):
    """Calculate Weight of Evidence and Information Value for a feature."""
    if df[feature].dtype in ['float64', 'int64']:
        df['binned'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df['binned'] = df[feature]

    grouped = df.groupby('binned')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    grouped['event_rate'] = grouped['events'] / total_events
    grouped['non_event_rate'] = grouped['non_events'] / total_non_events

    # Avoid division by zero
    grouped['event_rate'] = grouped['event_rate'].replace(0, 0.0001)
    grouped['non_event_rate'] = grouped['non_event_rate'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['non_event_rate'] / grouped['event_rate'])
    grouped['iv'] = (grouped['non_event_rate'] - grouped['event_rate']) * grouped['woe']

    iv = grouped['iv'].sum()
    df.drop('binned', axis=1, inplace=True)
    return iv, grouped


def woe_iv_analysis(df, target='default'):
    """Perform WOE/IV analysis for feature selection."""
    print("\n" + "=" * 60)
    print("WOE / INFORMATION VALUE ANALYSIS")
    print("=" * 60)

    numeric_features = ['annual_income', 'credit_score', 'loan_amount',
                        'employment_length', 'debt_to_income', 'num_credit_lines',
                        'delinquencies_2yr', 'interest_rate']

    iv_results = {}
    for feat in numeric_features:
        iv, _ = calculate_woe_iv(df.copy(), feat, target)
        iv_results[feat] = iv

    for feat in ['home_ownership', 'loan_purpose']:
        iv, _ = calculate_woe_iv(df.copy(), feat, target)
        iv_results[feat] = iv

    iv_df = pd.DataFrame({'Feature': list(iv_results.keys()), 'IV': list(iv_results.values())})
    iv_df = iv_df.sort_values('IV', ascending=False)
    iv_df['Predictive Power'] = iv_df['IV'].apply(
        lambda x: 'Strong' if x > 0.3 else ('Medium' if x > 0.1 else ('Weak' if x > 0.02 else 'Useless'))
    )

    print("\nInformation Value Ranking:")
    print(iv_df.to_string(index=False))
    return iv_df


def build_scorecard(model, scaler, feature_names, base_score=600, pdo=20):
    """Build a credit scorecard from logistic regression coefficients."""
    print("\n" + "=" * 60)
    print("SCORECARD DEVELOPMENT")
    print("=" * 60)

    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
    else:
        print("Scorecard requires logistic regression model.")
        return None

    factor = pdo / np.log(2)
    offset = base_score - factor * intercept

    scorecard = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Score_Points': (-factor * coefficients).round(1)
    })
    scorecard = scorecard.sort_values('Score_Points', ascending=False)

    print(f"\nBase Score: {base_score}, PDO: {pdo}")
    print(f"Score Factor: {factor:.2f}, Offset: {offset:.2f}")
    print("\nScorecard Points:")
    print(scorecard.to_string(index=False))

    return scorecard, factor, offset


def profit_curve_analysis(y_test, y_prob, loan_amounts_test):
    """Analyze profit curves for different approval thresholds."""
    print("\n" + "=" * 60)
    print("PROFIT CURVE ANALYSIS")
    print("=" * 60)

    avg_interest_earned = 0.08  # Average annual interest on approved loans
    avg_loss_on_default = 0.40  # Average loss given default

    thresholds = np.arange(0.01, 0.99, 0.01)
    profits = []

    for t in thresholds:
        approved = y_prob < t  # Approve if predicted default prob < threshold
        n_approved = approved.sum()

        approved_defaults = (y_test[approved] == 1).sum() if n_approved > 0 else 0
        approved_good = n_approved - approved_defaults

        profit_from_good = approved_good * np.mean(loan_amounts_test[approved]) * avg_interest_earned if approved_good > 0 else 0
        loss_from_defaults = approved_defaults * np.mean(loan_amounts_test[approved]) * avg_loss_on_default if approved_defaults > 0 else 0

        net_profit = profit_from_good - loss_from_defaults
        approval_rate = n_approved / len(y_test) * 100

        profits.append({
            'threshold': t, 'net_profit': net_profit,
            'approval_rate': approval_rate, 'n_approved': n_approved,
            'defaults_in_approved': approved_defaults
        })

    profits_df = pd.DataFrame(profits)
    optimal_idx = profits_df['net_profit'].idxmax()
    optimal = profits_df.loc[optimal_idx]

    print(f"\nOptimal approval threshold: {optimal['threshold']:.2f}")
    print(f"  Approval rate: {optimal['approval_rate']:.1f}%")
    print(f"  Net profit: ${optimal['net_profit']:,.0f}")
    print(f"  Defaults in approved: {optimal['defaults_in_approved']:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(profits_df['threshold'], profits_df['net_profit'], 'b-', linewidth=2)
    axes[0].axvline(x=optimal['threshold'], color='r', linestyle='--', label=f"Optimal ({optimal['threshold']:.2f})")
    axes[0].set_xlabel('Default Probability Threshold')
    axes[0].set_ylabel('Net Profit ($)')
    axes[0].set_title('Profit Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(profits_df['approval_rate'], profits_df['net_profit'], 'g-', linewidth=2)
    axes[1].set_xlabel('Approval Rate (%)')
    axes[1].set_ylabel('Net Profit ($)')
    axes[1].set_title('Profit vs Approval Rate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/profit_curves.png', dpi=150)
    plt.close()

    return profits_df


def fairness_assessment(df, y_test, y_pred, y_prob, test_indices):
    """Assess model fairness across demographic groups."""
    print("\n" + "=" * 60)
    print("MODEL FAIRNESS ASSESSMENT")
    print("=" * 60)

    test_df = df.iloc[test_indices].copy()
    test_df['y_true'] = y_test
    test_df['y_pred'] = y_pred
    test_df['y_prob'] = y_prob

    # Gender-based fairness
    print("\nFairness by Gender:")
    for gender in ['M', 'F']:
        subset = test_df[test_df['gender'] == gender]
        approval_rate = (subset['y_pred'] == 0).mean()
        actual_default = subset['y_true'].mean()
        avg_prob = subset['y_prob'].mean()
        print(f"  {gender}: Approval Rate={approval_rate:.3f}, "
              f"Actual Default Rate={actual_default:.3f}, "
              f"Avg Predicted Prob={avg_prob:.3f}")

    # Demographic parity check
    approval_rates = {}
    for gender in ['M', 'F']:
        subset = test_df[test_df['gender'] == gender]
        approval_rates[gender] = (subset['y_pred'] == 0).mean()

    disparity = abs(approval_rates['M'] - approval_rates['F'])
    print(f"\nDemographic Parity Disparity: {disparity:.4f}")
    if disparity < 0.05:
        print("  -> Fairness check PASSED (disparity < 5%)")
    else:
        print(f"  -> Fairness check WARNING (disparity >= 5%)")

    # Equal opportunity check (TPR parity)
    print("\nEqual Opportunity Check (among actual defaults):")
    for gender in ['M', 'F']:
        subset = test_df[(test_df['gender'] == gender) & (test_df['y_true'] == 1)]
        if len(subset) > 0:
            tpr = (subset['y_pred'] == 1).mean()
            print(f"  {gender}: True Positive Rate = {tpr:.3f}")


def main():
    """Main execution pipeline."""
    print("Loan Default Prediction Pipeline")
    print("=" * 60)

    # Generate data
    df = generate_loan_data(5000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.3f}")

    # WOE/IV Analysis
    woe_iv_analysis(df)

    # Prepare features
    df_encoded = pd.get_dummies(df, columns=['home_ownership', 'loan_purpose'], drop_first=True)
    feature_cols = [c for c in df_encoded.columns if c not in ['default', 'gender']]
    X = df_encoded[feature_cols].values
    y = df_encoded['default'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob, 'auc': auc, 'brier': brier}
        print(f"\n{name}:")
        print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test AUC: {auc:.4f}  |  Brier Score: {brier:.4f}")

    # Probability calibration
    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']

    calibrated_model = CalibratedClassifierCV(best_model, cv=5, method='isotonic')
    calibrated_model.fit(X_train_scaled, y_train)
    y_prob_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]

    brier_before = brier_score_loss(y_test, results[best_name]['y_prob'])
    brier_after = brier_score_loss(y_test, y_prob_calibrated)

    print(f"\n{best_name} - Brier Score:")
    print(f"  Before calibration: {brier_before:.4f}")
    print(f"  After calibration:  {brier_after:.4f}")

    # Calibration plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, res in results.items():
        fraction_pos, mean_pred = calibration_curve(y_test, res['y_prob'], n_bins=10)
        ax.plot(mean_pred, fraction_pos, 's-', label=f'{name}')

    fraction_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_calibrated, n_bins=10)
    ax.plot(mean_pred_cal, fraction_pos_cal, 'D--', label=f'{best_name} (Calibrated)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/calibration_curves.png', dpi=150)
    plt.close()

    # Scorecard
    lr_model = results['Logistic Regression']['model']
    build_scorecard(lr_model, scaler, feature_cols)

    # Profit curve analysis
    loan_amounts_test = df.iloc[idx_test]['loan_amount'].values
    profit_curve_analysis(y_test, results[best_name]['y_prob'], loan_amounts_test)

    # Fairness assessment
    fairness_assessment(df, y_test, results[best_name]['y_pred'],
                        results[best_name]['y_prob'], idx_test)

    # ROC comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Loan Default Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Pipeline complete. All results and visualizations saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
