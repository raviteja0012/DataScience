"""
Project 6: Medical Cost Prediction
====================================
Synthetic insurance cost prediction using multiple regression models.
Demonstrates EDA with seaborn, feature engineering, residual analysis,
and comprehensive model comparison.

Models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/user/DataScience/projects/01-machine-learning/06-medical-cost-prediction'


# =============================================================================
# 1. Synthetic Data Generation
# =============================================================================
def generate_insurance_data(n_samples=1500, random_state=42):
    """
    Generate synthetic medical insurance cost data.
    Charges depend on age, BMI, smoking status, children, and region.
    """
    np.random.seed(random_state)
    print("=" * 70)
    print("GENERATING SYNTHETIC INSURANCE DATA")
    print("=" * 70)

    # Demographics
    age = np.random.randint(18, 65, size=n_samples)
    sex = np.random.choice(['male', 'female'], size=n_samples)
    bmi = np.random.normal(30.0, 6.0, size=n_samples).clip(15, 55)
    children = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples,
                                 p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    smoker = np.random.choice(['yes', 'no'], size=n_samples, p=[0.20, 0.80])
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'],
                               size=n_samples)

    # Generate charges based on realistic relationships
    charges = np.zeros(n_samples)

    # Base charge from age (older = higher)
    charges += 250 * age + 0.5 * age ** 2

    # BMI effect (exponential above 30)
    bmi_effect = np.where(bmi > 30, 500 * (bmi - 30) ** 1.2, 100 * bmi)
    charges += bmi_effect

    # Smoking is the biggest factor
    smoker_binary = (smoker == 'yes').astype(float)
    charges += smoker_binary * 23000

    # Interaction: smoker + high BMI is devastating
    charges += smoker_binary * np.where(bmi > 30, 1500 * (bmi - 30), 0)

    # Children effect
    charges += children * 500

    # Region effect (small)
    region_effect = {'northeast': 500, 'northwest': 0, 'southeast': 1000, 'southwest': -200}
    charges += np.array([region_effect[r] for r in region])

    # Sex effect (small)
    charges += np.where(sex == 'male', 1300, 0)

    # Add noise
    charges += np.random.normal(0, 2000, size=n_samples)
    charges = np.maximum(charges, 1000)  # Minimum charge

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi.round(1),
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges.round(2)
    })

    print(f"Generated {n_samples} insurance records")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nBasic statistics:")
    print(df.describe().round(2).to_string())

    return df


# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================
def exploratory_data_analysis(df):
    """Comprehensive EDA with seaborn plots."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Categorical distributions
    for col in ['sex', 'smoker', 'region']:
        print(f"\n{col} distribution:")
        print(df[col].value_counts().to_string())

    # --- Plot 1: Distribution of charges ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    sns.histplot(df['charges'], bins=50, kde=True, ax=ax, color='#3498db')
    ax.set_title('Distribution of Medical Charges', fontsize=13, fontweight='bold')
    ax.set_xlabel('Charges ($)')
    ax.axvline(df['charges'].mean(), color='red', linestyle='--', label=f"Mean: ${df['charges'].mean():,.0f}")
    ax.axvline(df['charges'].median(), color='green', linestyle='--', label=f"Median: ${df['charges'].median():,.0f}")
    ax.legend()

    # Age vs Charges
    ax = axes[0, 1]
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker',
                    alpha=0.6, ax=ax, palette={'yes': '#e74c3c', 'no': '#3498db'})
    ax.set_title('Age vs Charges (by Smoking Status)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Charges ($)')

    # BMI vs Charges
    ax = axes[0, 2]
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker',
                    alpha=0.6, ax=ax, palette={'yes': '#e74c3c', 'no': '#3498db'})
    ax.set_title('BMI vs Charges (by Smoking Status)', fontsize=13, fontweight='bold')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Charges ($)')

    # Box plot: Smoker vs Charges
    ax = axes[1, 0]
    sns.boxplot(data=df, x='smoker', y='charges', ax=ax,
                palette={'yes': '#e74c3c', 'no': '#3498db'})
    ax.set_title('Charges by Smoking Status', fontsize=13, fontweight='bold')

    # Box plot: Region vs Charges
    ax = axes[1, 1]
    sns.boxplot(data=df, x='region', y='charges', ax=ax, palette='Set2')
    ax.set_title('Charges by Region', fontsize=13, fontweight='bold')

    # Children vs Charges
    ax = axes[1, 2]
    sns.boxplot(data=df, x='children', y='charges', ax=ax, palette='viridis')
    ax.set_title('Charges by Number of Children', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nEDA plots saved to eda_plots.png")

    # --- Plot 2: Correlation heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_numeric = df.copy()
    df_numeric['smoker_encoded'] = (df_numeric['smoker'] == 'yes').astype(int)
    df_numeric['sex_encoded'] = (df_numeric['sex'] == 'male').astype(int)
    numeric_cols = ['age', 'bmi', 'children', 'smoker_encoded', 'sex_encoded', 'charges']

    ax = axes[0]
    corr = df_numeric[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=ax, square=True)
    ax.set_title('Feature Correlations', fontsize=13, fontweight='bold')

    # Pairplot substitute: key relationships
    ax = axes[1]
    sns.violinplot(data=df, x='smoker', y='charges', hue='sex', split=True,
                   ax=ax, palette='Set1', inner='quartile')
    ax.set_title('Charges: Smoker x Sex', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Correlation plots saved to correlation_plots.png")

    # Key insights
    smoker_mean = df[df['smoker'] == 'yes']['charges'].mean()
    nonsmoker_mean = df[df['smoker'] == 'no']['charges'].mean()
    print(f"\nKey Insights:")
    print(f"  Average charges (smoker): ${smoker_mean:,.2f}")
    print(f"  Average charges (non-smoker): ${nonsmoker_mean:,.2f}")
    print(f"  Smoker premium: {smoker_mean / nonsmoker_mean:.1f}x higher")
    print(f"  Correlation(age, charges): {corr.loc['age', 'charges']:.3f}")
    print(f"  Correlation(bmi, charges): {corr.loc['bmi', 'charges']:.3f}")
    print(f"  Correlation(smoker, charges): {corr.loc['smoker_encoded', 'charges']:.3f}")


# =============================================================================
# 3. Feature Engineering
# =============================================================================
def feature_engineering(df):
    """Create engineered features for better model performance."""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df_eng = df.copy()

    # Encode categorical variables
    df_eng['smoker_binary'] = (df_eng['smoker'] == 'yes').astype(int)
    df_eng['sex_binary'] = (df_eng['sex'] == 'male').astype(int)

    # One-hot encode region
    region_dummies = pd.get_dummies(df_eng['region'], prefix='region', drop_first=True)
    df_eng = pd.concat([df_eng, region_dummies], axis=1)

    # Polynomial / interaction features
    df_eng['age_squared'] = df_eng['age'] ** 2
    df_eng['bmi_squared'] = df_eng['bmi'] ** 2
    df_eng['age_bmi'] = df_eng['age'] * df_eng['bmi']

    # BMI categories (clinical)
    df_eng['bmi_category'] = pd.cut(df_eng['bmi'],
                                     bins=[0, 18.5, 25, 30, 35, 100],
                                     labels=[0, 1, 2, 3, 4]).astype(int)

    # Smoker x BMI interaction (key driver)
    df_eng['smoker_bmi'] = df_eng['smoker_binary'] * df_eng['bmi']
    df_eng['smoker_bmi_over30'] = df_eng['smoker_binary'] * np.maximum(df_eng['bmi'] - 30, 0)

    # Age group
    df_eng['age_group'] = pd.cut(df_eng['age'],
                                  bins=[17, 30, 45, 55, 65],
                                  labels=[0, 1, 2, 3]).astype(int)

    # Smoker x Age interaction
    df_eng['smoker_age'] = df_eng['smoker_binary'] * df_eng['age']

    # Log transform of charges (for potential log-linear model)
    df_eng['log_charges'] = np.log1p(df_eng['charges'])

    # Drop original categorical columns
    df_eng.drop(['sex', 'smoker', 'region'], axis=1, inplace=True)

    feature_cols = [c for c in df_eng.columns if c not in ['charges', 'log_charges']]
    print(f"Engineered features ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    return df_eng, feature_cols


# =============================================================================
# 4. Model Training
# =============================================================================
def train_models(df_eng, feature_cols):
    """Train multiple regression models."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    X = df_eng[feature_cols]
    y = df_eng['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    models = {}
    results = {}

    # --- Model 1: Linear Regression ---
    print("\n--- Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2')
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'y_pred': y_pred_lr,
        'r2': r2_score(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    print(f"  R2: {results['Linear Regression']['r2']:.4f}")
    print(f"  RMSE: ${results['Linear Regression']['rmse']:,.2f}")
    print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- Model 2: Ridge Regression ---
    print("\n--- Ridge Regression ---")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
    models['Ridge'] = ridge
    results['Ridge'] = {
        'y_pred': y_pred_ridge,
        'r2': r2_score(y_test, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    print(f"  R2: {results['Ridge']['r2']:.4f}")
    print(f"  RMSE: ${results['Ridge']['rmse']:,.2f}")
    print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- Model 3: Lasso Regression ---
    print("\n--- Lasso Regression ---")
    lasso = Lasso(alpha=50.0, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    cv_scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='r2')
    models['Lasso'] = lasso
    results['Lasso'] = {
        'y_pred': y_pred_lasso,
        'r2': r2_score(y_test, y_pred_lasso),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        'mae': mean_absolute_error(y_test, y_pred_lasso),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    # Features selected by Lasso (non-zero coefficients)
    n_selected = np.sum(lasso.coef_ != 0)
    print(f"  R2: {results['Lasso']['r2']:.4f}")
    print(f"  RMSE: ${results['Lasso']['rmse']:,.2f}")
    print(f"  Features selected: {n_selected}/{len(feature_cols)}")
    print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- Model 4: Random Forest ---
    print("\n--- Random Forest ---")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # No scaling needed for tree models
    y_pred_rf = rf.predict(X_test)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'y_pred': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    print(f"  R2: {results['Random Forest']['r2']:.4f}")
    print(f"  RMSE: ${results['Random Forest']['rmse']:,.2f}")
    print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- Model 5: Gradient Boosting ---
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        min_samples_split=5, subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='r2')
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'y_pred': y_pred_gb,
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
    print(f"  R2: {results['Gradient Boosting']['r2']:.4f}")
    print(f"  RMSE: ${results['Gradient Boosting']['rmse']:,.2f}")
    print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return models, results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# =============================================================================
# 5. Model Comparison
# =============================================================================
def model_comparison(results):
    """Compare all models in a summary table and plot."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            'Model': name,
            'R2': res['r2'],
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'CV R2 (mean)': res['cv_r2_mean'],
            'CV R2 (std)': res['cv_r2_std'],
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('R2', ascending=False)
    print("\n" + comparison_df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = comparison_df['Model'].values
    colors = sns.color_palette('viridis', len(model_names))

    # R2 Score
    ax = axes[0]
    bars = ax.barh(model_names, comparison_df['R2'].values, color=colors)
    ax.set_xlabel('R2 Score')
    ax.set_title('R2 Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, comparison_df['R2'].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=10)

    # RMSE
    ax = axes[1]
    bars = ax.barh(model_names, comparison_df['RMSE'].values, color=colors)
    ax.set_xlabel('RMSE ($)')
    ax.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, comparison_df['RMSE'].values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
                va='center', fontsize=10)

    # CV R2 with error bars
    ax = axes[2]
    ax.barh(model_names, comparison_df['CV R2 (mean)'].values, color=colors,
            xerr=comparison_df['CV R2 (std)'].values, capsize=5)
    ax.set_xlabel('CV R2 Score')
    ax.set_title('Cross-Validated R2 (5-Fold)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nModel comparison plots saved to model_comparison.png")

    return comparison_df


# =============================================================================
# 6. Residual Analysis
# =============================================================================
def residual_analysis(results, y_test):
    """Perform residual analysis on all models."""
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        if idx >= 5:
            break
        ax = axes_flat[idx]
        residuals = y_test.values - res['y_pred']

        ax.scatter(res['y_pred'], residuals, alpha=0.5, s=15, color='#3498db')
        ax.axhline(y=0, color='red', linestyle='--', lw=1)
        ax.set_xlabel('Predicted Charges ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title(f'{name}\nR2={res["r2"]:.4f}, RMSE=${res["rmse"]:,.0f}',
                     fontsize=11, fontweight='bold')

        # Print residual statistics
        print(f"\n{name} residuals:")
        print(f"  Mean: ${residuals.mean():,.2f}")
        print(f"  Std: ${residuals.std():,.2f}")
        print(f"  Median: ${np.median(residuals):,.2f}")
        print(f"  Max absolute: ${np.abs(residuals).max():,.2f}")

    # Last subplot: residual distribution for best model
    best_model = max(results, key=lambda k: results[k]['r2'])
    ax = axes_flat[5]
    best_residuals = y_test.values - results[best_model]['y_pred']
    sns.histplot(best_residuals, bins=50, kde=True, ax=ax, color='#2ecc71')
    ax.set_title(f'Residual Distribution ({best_model})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Residual ($)')
    ax.axvline(x=0, color='red', linestyle='--', lw=1)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nResidual analysis plots saved to residual_analysis.png")

    # Actual vs Predicted plot for best model
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, results[best_model]['y_pred'], alpha=0.5, s=15, color='#3498db')
    min_val = min(y_test.min(), results[best_model]['y_pred'].min())
    max_val = max(y_test.max(), results[best_model]['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Charges ($)', fontsize=12)
    ax.set_ylabel('Predicted Charges ($)', fontsize=12)
    ax.set_title(f'Actual vs Predicted ({best_model})\nR2={results[best_model]["r2"]:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Actual vs Predicted plot saved to actual_vs_predicted.png")


# =============================================================================
# 7. Feature Importance
# =============================================================================
def analyze_feature_importance(models, feature_cols):
    """Analyze feature importance from different models."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Linear Regression coefficients
    ax = axes[0]
    lr_coefs = models['Linear Regression'].coef_
    indices = np.argsort(np.abs(lr_coefs))[::-1][:10]
    colors = ['#e74c3c' if lr_coefs[i] < 0 else '#2ecc71' for i in indices]
    ax.barh([feature_cols[i] for i in indices], lr_coefs[indices], color=colors)
    ax.set_title('Linear Regression Coefficients (Top 10)', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='k', lw=0.5)

    # Random Forest importance
    ax = axes[1]
    rf_imp = models['Random Forest'].feature_importances_
    indices = np.argsort(rf_imp)[::-1][:10]
    ax.barh([feature_cols[i] for i in indices], rf_imp[indices], color='#3498db')
    ax.set_title('Random Forest Feature Importance (Top 10)', fontsize=11, fontweight='bold')

    # Gradient Boosting importance
    ax = axes[2]
    gb_imp = models['Gradient Boosting'].feature_importances_
    indices = np.argsort(gb_imp)[::-1][:10]
    ax.barh([feature_cols[i] for i in indices], gb_imp[indices], color='#e67e22')
    ax.set_title('Gradient Boosting Feature Importance (Top 10)', fontsize=11, fontweight='bold')

    for ax in axes:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance plots saved to feature_importance.png")

    # Lasso feature selection
    lasso_coefs = models['Lasso'].coef_
    selected = [(feature_cols[i], lasso_coefs[i]) for i in range(len(lasso_coefs)) if lasso_coefs[i] != 0]
    eliminated = [feature_cols[i] for i in range(len(lasso_coefs)) if lasso_coefs[i] == 0]

    print(f"\nLasso selected features ({len(selected)}):")
    for name, coef in sorted(selected, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.2f}")

    if eliminated:
        print(f"\nLasso eliminated features ({len(eliminated)}):")
        for name in eliminated:
            print(f"  {name}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  MEDICAL COST PREDICTION")
    print("  Insurance Charges Regression Analysis")
    print("=" * 70)

    # 1. Generate data
    df = generate_insurance_data(n_samples=1500, random_state=42)

    # 2. EDA
    exploratory_data_analysis(df)

    # 3. Feature engineering
    df_eng, feature_cols = feature_engineering(df)

    # 4. Train models
    models, results, X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = train_models(df_eng, feature_cols)

    # 5. Model comparison
    comparison_df = model_comparison(results)

    # 6. Residual analysis
    residual_analysis(results, y_test)

    # 7. Feature importance
    analyze_feature_importance(models, feature_cols)

    # Final summary
    best_model = comparison_df.iloc[0]
    print("\n" + "=" * 70)
    print("  MEDICAL COST PREDICTION COMPLETE")
    print("=" * 70)
    print(f"\n  Best model: {best_model['Model']}")
    print(f"  R2 Score: {best_model['R2']:.4f}")
    print(f"  RMSE: ${best_model['RMSE']:,.2f}")
    print(f"  MAE: ${best_model['MAE']:,.2f}")
    print(f"\nKey findings:")
    print(f"  - Smoking status is the strongest predictor of medical costs")
    print(f"  - BMI interacts with smoking to amplify costs")
    print(f"  - Age has a non-linear (quadratic) relationship with charges")
    print(f"  - Tree-based models capture interactions automatically")


if __name__ == '__main__':
    main()
