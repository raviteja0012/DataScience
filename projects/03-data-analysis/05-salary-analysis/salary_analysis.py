#!/usr/bin/env python3
"""
Salary Data Analysis
====================
Generates synthetic salary data and performs comprehensive analysis
including distribution analysis, regression modeling, and statistical
significance testing across demographics and job characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_RECORDS = 1000
OUTPUT_DIR = "outputs"

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_salary_data(n: int = N_RECORDS) -> pd.DataFrame:
    """Create a synthetic salary dataset with realistic relationships."""

    job_titles = [
        "Software Engineer", "Data Scientist", "Product Manager",
        "Marketing Manager", "Sales Representative", "Financial Analyst",
        "UX Designer", "DevOps Engineer", "Business Analyst",
        "HR Specialist", "Project Manager", "QA Engineer",
        "Content Writer", "Customer Support", "Administrative Assistant",
    ]
    job_weights = np.array([14, 10, 8, 7, 8, 7, 6, 6, 7, 5, 7, 5, 4, 4, 2],
                           dtype=float)
    job_weights /= job_weights.sum()

    # Base salary by job title (mean, std)
    salary_params = {
        "Software Engineer":       (110000, 22000),
        "Data Scientist":          (115000, 25000),
        "Product Manager":         (120000, 28000),
        "Marketing Manager":       (85000, 18000),
        "Sales Representative":    (60000, 15000),
        "Financial Analyst":       (80000, 17000),
        "UX Designer":             (90000, 19000),
        "DevOps Engineer":         (105000, 21000),
        "Business Analyst":        (78000, 16000),
        "HR Specialist":           (62000, 13000),
        "Project Manager":         (95000, 20000),
        "QA Engineer":             (82000, 17000),
        "Content Writer":          (55000, 12000),
        "Customer Support":        (42000, 9000),
        "Administrative Assistant": (40000, 8000),
    }

    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    edu_weights = np.array([8, 48, 32, 12], dtype=float)
    edu_weights /= edu_weights.sum()

    # Education premium multiplier
    edu_multiplier = {
        "High School": 0.82,
        "Bachelor's": 1.00,
        "Master's": 1.15,
        "PhD": 1.28,
    }

    locations = [
        "San Francisco", "New York", "Seattle", "Austin", "Chicago",
        "Boston", "Denver", "Los Angeles", "Atlanta", "Remote",
    ]
    loc_weights = np.array([15, 14, 10, 10, 8, 8, 7, 10, 8, 10], dtype=float)
    loc_weights /= loc_weights.sum()

    # Location cost-of-living multiplier
    loc_multiplier = {
        "San Francisco": 1.25, "New York": 1.22, "Seattle": 1.15,
        "Austin": 1.00, "Chicago": 1.02, "Boston": 1.18,
        "Denver": 1.05, "Los Angeles": 1.18, "Atlanta": 0.95,
        "Remote": 1.00,
    }

    genders = ["Male", "Female", "Non-Binary"]
    gender_weights = np.array([0.50, 0.45, 0.05])

    job_title = np.random.choice(job_titles, size=n, p=job_weights)
    education = np.random.choice(education_levels, size=n, p=edu_weights)
    location = np.random.choice(locations, size=n, p=loc_weights)
    gender = np.random.choice(genders, size=n, p=gender_weights)

    # Experience years (0-30, correlated with education)
    experience = []
    for edu in education:
        if edu == "PhD":
            experience.append(max(0, int(np.random.normal(12, 5))))
        elif edu == "Master's":
            experience.append(max(0, int(np.random.normal(8, 4))))
        elif edu == "Bachelor's":
            experience.append(max(0, int(np.random.normal(6, 4))))
        else:
            experience.append(max(0, int(np.random.normal(10, 6))))
    experience = np.clip(experience, 0, 30)

    # Age
    age = np.clip(
        np.array(experience) + np.random.randint(22, 28, n),
        22, 65,
    )

    # Calculate salary
    salaries = []
    for i in range(n):
        base_mean, base_std = salary_params[job_title[i]]
        base = np.random.normal(base_mean, base_std)

        # Apply multipliers
        salary = base * edu_multiplier[education[i]] * loc_multiplier[location[i]]

        # Experience premium: ~3% per year with diminishing returns
        exp_premium = 1 + 0.03 * experience[i] * (1 - experience[i] / 80)
        salary *= exp_premium

        # Small random gender gap (unfortunately realistic)
        if gender[i] == "Female":
            salary *= np.random.normal(0.95, 0.02)
        elif gender[i] == "Non-Binary":
            salary *= np.random.normal(0.96, 0.03)

        salaries.append(max(28000, int(salary)))

    # Bonus percentage (5-25% based on role and performance)
    bonus_pct = np.clip(np.random.normal(12, 5, n), 0, 35).round(1)

    # Satisfaction (1-5)
    satisfaction = np.clip(np.random.normal(3.6, 0.8, n), 1, 5).round(1)

    # Remote work percentage
    remote_pct = np.where(
        np.array(location) == "Remote",
        100,
        np.clip(np.random.normal(40, 25, n), 0, 100).astype(int),
    )

    df = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "job_title": job_title,
        "experience_years": experience,
        "education": education,
        "location": location,
        "gender": gender,
        "age": age,
        "salary": salaries,
        "bonus_pct": bonus_pct,
        "satisfaction": satisfaction,
        "remote_pct": remote_pct,
    })

    df["total_compensation"] = (df["salary"] * (1 + df["bonus_pct"] / 100)).astype(int)

    return df


# ===========================================================================
# 2. Exploratory Data Analysis
# ===========================================================================
def basic_overview(df: pd.DataFrame) -> None:
    """Print dataset summary statistics."""
    print("=" * 70)
    print("SALARY DATASET — BASIC OVERVIEW")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"\n--- Descriptive Statistics ---")
    print(df.describe().round(1))
    print(f"\n--- Job Title Distribution ---")
    print(df["job_title"].value_counts())
    print(f"\n--- Education Distribution ---")
    print(df["education"].value_counts())
    print()


def salary_distribution_analysis(df: pd.DataFrame) -> None:
    """Visualise salary distributions across dimensions."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Salary Distribution Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Overall salary histogram
    sns.histplot(data=df, x="salary", bins=30, kde=True, color="steelblue", ax=axes[0, 0])
    axes[0, 0].axvline(df["salary"].mean(), color="red", linewidth=1.5,
                       linestyle="--", label=f"Mean: ${df['salary'].mean():,.0f}")
    axes[0, 0].axvline(df["salary"].median(), color="green", linewidth=1.5,
                       linestyle="--", label=f"Median: ${df['salary'].median():,.0f}")
    axes[0, 0].set_title("Salary Distribution", fontsize=14)
    axes[0, 0].legend(fontsize=9)

    # 2. Salary by job title
    job_order = df.groupby("job_title")["salary"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y="job_title", x="salary", order=job_order,
                palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("Salary by Job Title", fontsize=14)
    axes[0, 1].set_ylabel("")

    # 3. Salary by education
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    sns.violinplot(data=df, x="education", y="salary", order=edu_order,
                   palette="muted", inner="quartile", ax=axes[0, 2])
    axes[0, 2].set_title("Salary by Education", fontsize=14)

    # 4. Salary by location
    loc_order = df.groupby("location")["salary"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y="location", x="salary", order=loc_order,
                palette="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title("Salary by Location", fontsize=14)
    axes[1, 0].set_ylabel("")

    # 5. Salary by gender
    sns.boxplot(data=df, x="gender", y="salary", palette="Set2", ax=axes[1, 1])
    axes[1, 1].set_title("Salary by Gender", fontsize=14)

    # Gender means annotated
    for i, g in enumerate(["Female", "Male", "Non-Binary"]):
        mean_val = df.loc[df["gender"] == g, "salary"].mean()
        axes[1, 1].text(i, mean_val + 3000, f"${mean_val:,.0f}",
                        ha="center", fontsize=9, color="red")

    # 6. Salary vs Experience scatter
    sns.scatterplot(data=df, x="experience_years", y="salary",
                    hue="education", alpha=0.5, ax=axes[1, 2])
    axes[1, 2].set_title("Salary vs Experience (by Education)", fontsize=14)
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/salary_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] salary_distributions.png")


def regression_analysis(df: pd.DataFrame) -> None:
    """Perform and visualise regression analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Regression Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Simple linear regression: salary vs experience
    x = df["experience_years"].values
    y = df["salary"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    axes[0, 0].scatter(x, y, alpha=0.3, color="steelblue", s=20)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0, 0].plot(x_line, slope * x_line + intercept, color="red",
                    linewidth=2, label=f"y = {slope:.0f}x + {intercept:.0f}\nR² = {r_value**2:.3f}")
    axes[0, 0].set_title("Salary vs Experience (Linear Regression)", fontsize=14)
    axes[0, 0].set_xlabel("Experience (years)")
    axes[0, 0].set_ylabel("Salary ($)")
    axes[0, 0].legend()

    print(f"\n--- Simple Linear Regression: Salary ~ Experience ---")
    print(f"  Slope     : {slope:,.2f} (per year of experience)")
    print(f"  Intercept : ${intercept:,.0f}")
    print(f"  R-squared : {r_value**2:.4f}")
    print(f"  p-value   : {p_value:.6e}")
    print(f"  Std Error : {std_err:.2f}")

    # 2. Residual plot
    predicted = slope * x + intercept
    residuals = y - predicted
    axes[0, 1].scatter(predicted, residuals, alpha=0.3, color="coral", s=20)
    axes[0, 1].axhline(y=0, color="black", linewidth=1, linestyle="--")
    axes[0, 1].set_title("Residual Plot", fontsize=14)
    axes[0, 1].set_xlabel("Predicted Salary")
    axes[0, 1].set_ylabel("Residual")

    # 3. Multiple regression using encoded features
    df_reg = df.copy()
    edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df_reg["edu_encoded"] = df_reg["education"].map(edu_map)

    # Manual multiple regression (experience + education)
    X = np.column_stack([
        df_reg["experience_years"].values,
        df_reg["edu_encoded"].values,
        np.ones(len(df_reg)),
    ])
    y_vec = df_reg["salary"].values
    # OLS: beta = (X'X)^-1 X'y
    beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
    y_pred_multi = X @ beta
    ss_res = np.sum((y_vec - y_pred_multi) ** 2)
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    r2_multi = 1 - ss_res / ss_tot

    print(f"\n--- Multiple Regression: Salary ~ Experience + Education ---")
    print(f"  Beta (experience) : {beta[0]:,.2f}")
    print(f"  Beta (education)  : {beta[1]:,.2f}")
    print(f"  Intercept         : ${beta[2]:,.0f}")
    print(f"  R-squared         : {r2_multi:.4f}")

    # Predicted vs actual
    axes[1, 0].scatter(y_vec, y_pred_multi, alpha=0.3, color="seagreen", s=20)
    min_val = min(y_vec.min(), y_pred_multi.min())
    max_val = max(y_vec.max(), y_pred_multi.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    axes[1, 0].set_title(f"Predicted vs Actual (R² = {r2_multi:.3f})", fontsize=14)
    axes[1, 0].set_xlabel("Actual Salary ($)")
    axes[1, 0].set_ylabel("Predicted Salary ($)")

    # 4. Experience vs salary by education with regression lines
    for edu in ["High School", "Bachelor's", "Master's", "PhD"]:
        subset = df[df["education"] == edu]
        sns.regplot(data=subset, x="experience_years", y="salary",
                    scatter_kws={"alpha": 0.3, "s": 15}, label=edu,
                    ax=axes[1, 1], ci=None)
    axes[1, 1].set_title("Regression Lines by Education Level", fontsize=14)
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/regression_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] regression_analysis.png")


def statistical_significance_tests(df: pd.DataFrame) -> None:
    """Comprehensive statistical hypothesis testing."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    # 1. ANOVA: Salary across education levels
    groups_edu = [grp["salary"].values for _, grp in df.groupby("education")]
    f_stat, p_val = stats.f_oneway(*groups_edu)
    print(f"\n1. One-Way ANOVA — Salary across Education Levels")
    print(f"   F = {f_stat:.4f}, p-value = {p_val:.6e}")
    print(f"   {'Significant' if p_val < 0.05 else 'Not significant'} at alpha = 0.05")

    # 2. ANOVA: Salary across job titles
    groups_job = [grp["salary"].values for _, grp in df.groupby("job_title")]
    f_job, p_job = stats.f_oneway(*groups_job)
    print(f"\n2. One-Way ANOVA — Salary across Job Titles")
    print(f"   F = {f_job:.4f}, p-value = {p_job:.6e}")
    print(f"   {'Significant' if p_job < 0.05 else 'Not significant'} at alpha = 0.05")

    # 3. t-test: Male vs Female salary
    male_salary = df.loc[df["gender"] == "Male", "salary"]
    female_salary = df.loc[df["gender"] == "Female", "salary"]
    t_stat, p_gender = stats.ttest_ind(male_salary, female_salary, equal_var=False)
    print(f"\n3. Welch's t-Test — Male vs Female Salary")
    print(f"   Mean Male = ${male_salary.mean():,.0f}, Mean Female = ${female_salary.mean():,.0f}")
    print(f"   Difference = ${male_salary.mean() - female_salary.mean():,.0f}")
    print(f"   t = {t_stat:.4f}, p-value = {p_gender:.6f}")
    print(f"   {'Significant' if p_gender < 0.05 else 'Not significant'} at alpha = 0.05")

    # 4. ANOVA: Salary across locations
    groups_loc = [grp["salary"].values for _, grp in df.groupby("location")]
    f_loc, p_loc = stats.f_oneway(*groups_loc)
    print(f"\n4. One-Way ANOVA — Salary across Locations")
    print(f"   F = {f_loc:.4f}, p-value = {p_loc:.6e}")
    print(f"   {'Significant' if p_loc < 0.05 else 'Not significant'} at alpha = 0.05")

    # 5. Pearson correlation: Experience vs Salary
    r_exp, p_exp = stats.pearsonr(df["experience_years"], df["salary"])
    print(f"\n5. Pearson Correlation — Experience vs Salary")
    print(f"   r = {r_exp:.4f}, p-value = {p_exp:.6e}")
    print(f"   {'Significant' if p_exp < 0.05 else 'Not significant'} at alpha = 0.05")

    # 6. Spearman correlation: Education (ordinal) vs Salary
    edu_ord = df["education"].map({"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3})
    rho, p_rho = stats.spearmanr(edu_ord, df["salary"])
    print(f"\n6. Spearman Correlation — Education Level vs Salary")
    print(f"   rho = {rho:.4f}, p-value = {p_rho:.6e}")
    print(f"   {'Significant' if p_rho < 0.05 else 'Not significant'} at alpha = 0.05")

    # 7. Kruskal-Wallis: Non-parametric test across education
    h_stat, p_kw = stats.kruskal(*groups_edu)
    print(f"\n7. Kruskal-Wallis Test — Salary across Education Levels")
    print(f"   H = {h_stat:.4f}, p-value = {p_kw:.6e}")
    print(f"   {'Significant' if p_kw < 0.05 else 'Not significant'} at alpha = 0.05")

    # 8. Normality test on salary
    if len(df) <= 5000:
        stat_norm, p_norm = stats.shapiro(df["salary"].sample(min(500, len(df)),
                                                               random_state=SEED))
        print(f"\n8. Shapiro-Wilk Normality Test — Salary")
        print(f"   W = {stat_norm:.4f}, p-value = {p_norm:.6e}")
        print(f"   {'Normal' if p_norm > 0.05 else 'Not normal'} at alpha = 0.05")

    print("=" * 70)


def correlation_and_heatmap(df: pd.DataFrame) -> None:
    """Correlation analysis and heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Correlation Analysis", fontsize=18, fontweight="bold", y=1.02)

    # Encode categoricals for correlation
    df_encoded = df.copy()
    df_encoded["edu_encoded"] = df_encoded["education"].map(
        {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3})

    numeric = df_encoded[["experience_years", "age", "salary", "bonus_pct",
                          "satisfaction", "remote_pct", "total_compensation",
                          "edu_encoded"]]
    corr = numeric.corr()

    # 1. Heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=axes[0])
    axes[0].set_title("Correlation Matrix", fontsize=14)

    # 2. Pairplot-style: key variable pairs
    pairs = [
        ("experience_years", "salary"),
        ("edu_encoded", "salary"),
        ("age", "salary"),
        ("satisfaction", "salary"),
    ]
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    for idx, (x_col, y_col) in enumerate(pairs):
        r, _ = stats.pearsonr(df_encoded[x_col], df_encoded[y_col])
        axes[1].scatter(df_encoded[x_col] + idx * 0.1,
                        df_encoded[y_col],
                        alpha=0.1, s=10, color=colors[idx])
        axes[1].text(0.02, 0.95 - idx * 0.06,
                     f"{x_col} vs {y_col}: r={r:.3f}",
                     transform=axes[1].transAxes, fontsize=9,
                     color=colors[idx], fontweight="bold")
    axes[1].set_title("Correlation Summaries", fontsize=14)
    axes[1].set_ylabel("Salary ($)")
    axes[1].set_xlabel("Feature values (overlaid)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] correlation_analysis.png")


def compensation_deep_dive(df: pd.DataFrame) -> None:
    """Total compensation analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Compensation Deep Dive", fontsize=18, fontweight="bold", y=1.02)

    # 1. Total comp distribution
    sns.histplot(data=df, x="total_compensation", bins=30, kde=True,
                 color="darkgreen", ax=axes[0])
    axes[0].set_title("Total Compensation Distribution", fontsize=14)
    axes[0].set_xlabel("Total Compensation ($)")

    # 2. Bonus % by job title
    job_order = df.groupby("job_title")["bonus_pct"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y="job_title", x="bonus_pct", order=job_order,
                palette="YlOrBr", ax=axes[1])
    axes[1].set_title("Bonus % by Job Title", fontsize=14)
    axes[1].set_ylabel("")

    # 3. Salary vs Satisfaction
    sns.scatterplot(data=df, x="salary", y="satisfaction", hue="gender",
                    alpha=0.5, ax=axes[2])
    axes[2].set_title("Salary vs Satisfaction", fontsize=14)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/compensation_deep_dive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] compensation_deep_dive.png")


def print_key_insights(df: pd.DataFrame) -> None:
    """Print key findings."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(f"  Total records             : {len(df):,}")
    print(f"  Mean salary               : ${df['salary'].mean():,.0f}")
    print(f"  Median salary             : ${df['salary'].median():,.0f}")
    print(f"  Salary std dev            : ${df['salary'].std():,.0f}")
    print(f"  Salary range              : ${df['salary'].min():,} – ${df['salary'].max():,}")

    top_job = df.groupby("job_title")["salary"].mean().idxmax()
    top_salary = df.groupby("job_title")["salary"].mean().max()
    print(f"  Highest-paying title      : {top_job} (${top_salary:,.0f})")

    top_loc = df.groupby("location")["salary"].mean().idxmax()
    print(f"  Highest-paying location   : {top_loc}")

    male_mean = df.loc[df["gender"] == "Male", "salary"].mean()
    female_mean = df.loc[df["gender"] == "Female", "salary"].mean()
    gap = (male_mean - female_mean) / male_mean * 100
    print(f"  Gender pay gap (M vs F)   : {gap:.1f}%")

    r, _ = stats.pearsonr(df["experience_years"], df["salary"])
    print(f"  Experience-salary corr    : {r:.3f}")

    print(f"  Mean total compensation   : ${df['total_compensation'].mean():,.0f}")
    print(f"  Mean bonus %              : {df['bonus_pct'].mean():.1f}%")
    print("=" * 70)


# ===========================================================================
# 3. Main
# ===========================================================================
def main() -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic salary dataset ...")
    df = generate_salary_data()
    df.to_csv(f"{OUTPUT_DIR}/salary_synthetic.csv", index=False)
    print(f"[Saved] salary_synthetic.csv  ({len(df)} rows)\n")

    basic_overview(df)
    salary_distribution_analysis(df)
    regression_analysis(df)
    statistical_significance_tests(df)
    correlation_and_heatmap(df)
    compensation_deep_dive(df)
    print_key_insights(df)

    print("\nAll analyses complete. Charts saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
