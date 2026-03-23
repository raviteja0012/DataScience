#!/usr/bin/env python3
"""
HR Analytics Dashboard
======================
Generates synthetic HR employee data and performs comprehensive analytics
including attrition analysis, salary equity assessment, and statistical tests.
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
N_EMPLOYEES = 500
OUTPUT_DIR = "outputs"

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_hr_data(n: int = N_EMPLOYEES) -> pd.DataFrame:
    """Create a synthetic HR dataset with realistic distributions."""

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance",
                   "Operations", "Support", "Legal"]
    dept_weights = np.array([25, 20, 12, 8, 10, 12, 8, 5], dtype=float)
    dept_weights /= dept_weights.sum()
    department = np.random.choice(departments, size=n, p=dept_weights)

    # Base salary depends on department
    salary_map = {
        "Engineering": (85000, 20000),
        "Sales": (65000, 18000),
        "Marketing": (62000, 15000),
        "HR": (58000, 12000),
        "Finance": (75000, 18000),
        "Operations": (55000, 14000),
        "Support": (48000, 10000),
        "Legal": (80000, 20000),
    }

    salaries = []
    for d in department:
        mu, sigma = salary_map[d]
        salaries.append(max(30000, int(np.random.normal(mu, sigma))))

    gender = np.random.choice(["Male", "Female", "Non-Binary"], size=n,
                              p=[0.50, 0.45, 0.05])

    # Age: 22–60
    age = np.clip(np.random.normal(35, 8, n), 22, 60).astype(int)

    # Tenure in years (0–25)
    tenure = np.clip(np.random.exponential(4, n), 0, 25).round(1)

    # Satisfaction score 1–5
    satisfaction = np.clip(np.random.normal(3.4, 0.9, n), 1, 5).round(1)

    # Performance score 1–5
    performance = np.clip(np.random.normal(3.5, 0.7, n), 1, 5).round(1)

    # Work-life balance 1–5
    work_life_balance = np.clip(np.random.normal(3.2, 0.8, n), 1, 5).round(1)

    # Monthly hours
    monthly_hours = np.clip(np.random.normal(170, 20, n), 130, 230).astype(int)

    # Number of projects
    num_projects = np.random.poisson(4, n).clip(1, 10)

    # Promotion in last 3 years
    promoted = np.random.choice([0, 1], size=n, p=[0.82, 0.18])

    # Attrition: influenced by satisfaction, salary, work-life balance
    attrition_prob = (
        0.10
        + 0.15 * (5 - satisfaction) / 4
        + 0.10 * (5 - work_life_balance) / 4
        + 0.08 * (monthly_hours - 160) / 70
        - 0.05 * promoted
    )
    attrition_prob = np.clip(attrition_prob, 0.02, 0.70)
    attrition = np.random.binomial(1, attrition_prob)

    education = np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD"],
        size=n, p=[0.10, 0.50, 0.30, 0.10],
    )

    df = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": department,
        "gender": gender,
        "age": age,
        "education": education,
        "tenure_years": tenure,
        "salary": salaries,
        "satisfaction": satisfaction,
        "performance": performance,
        "work_life_balance": work_life_balance,
        "monthly_hours": monthly_hours,
        "num_projects": num_projects,
        "promoted_last_3yr": promoted,
        "attrition": attrition,
    })

    return df


# ===========================================================================
# 2. Exploratory Data Analysis
# ===========================================================================
def basic_overview(df: pd.DataFrame) -> None:
    """Print dataset summary."""
    print("=" * 70)
    print("HR ANALYTICS — BASIC OVERVIEW")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"\nAttrition rate: {df['attrition'].mean():.1%}")
    print(f"\n--- Descriptive Statistics ---")
    print(df.describe().round(2))
    print(f"\n--- Department Counts ---")
    print(df["department"].value_counts())
    print()


def attrition_analysis(df: pd.DataFrame) -> None:
    """Deep-dive into employee attrition patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Attrition Analysis", fontsize=18, fontweight="bold", y=1.02)

    attrition_label = df["attrition"].map({0: "Stayed", 1: "Left"})

    # 1. Overall attrition
    counts = attrition_label.value_counts()
    axes[0, 0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=["#66c2a5", "#fc8d62"], startangle=90,
                   textprops={"fontsize": 13})
    axes[0, 0].set_title("Overall Attrition", fontsize=14)

    # 2. Attrition by department
    dept_attr = df.groupby("department")["attrition"].mean().sort_values(ascending=False)
    sns.barplot(x=dept_attr.values, y=dept_attr.index, palette="Reds_r", ax=axes[0, 1])
    axes[0, 1].set_title("Attrition Rate by Department", fontsize=14)
    axes[0, 1].set_xlabel("Attrition Rate")
    for i, v in enumerate(dept_attr.values):
        axes[0, 1].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=10)

    # 3. Satisfaction vs attrition
    sns.boxplot(data=df, x="attrition", y="satisfaction", palette="Set2", ax=axes[0, 2])
    axes[0, 2].set_xticklabels(["Stayed", "Left"])
    axes[0, 2].set_title("Satisfaction by Attrition", fontsize=14)

    # 4. Monthly hours vs attrition
    sns.violinplot(data=df, x="attrition", y="monthly_hours", palette="Set3",
                   inner="quartile", ax=axes[1, 0])
    axes[1, 0].set_xticklabels(["Stayed", "Left"])
    axes[1, 0].set_title("Monthly Hours by Attrition", fontsize=14)

    # 5. Tenure vs attrition
    sns.histplot(data=df, x="tenure_years", hue=attrition_label, bins=15,
                 kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Tenure Distribution by Attrition", fontsize=14)

    # 6. Projects vs attrition
    proj_attr = df.groupby("num_projects")["attrition"].mean()
    axes[1, 2].bar(proj_attr.index, proj_attr.values, color="steelblue")
    axes[1, 2].set_title("Attrition Rate by # Projects", fontsize=14)
    axes[1, 2].set_xlabel("Number of Projects")
    axes[1, 2].set_ylabel("Attrition Rate")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attrition_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] attrition_analysis.png")


def salary_equity_analysis(df: pd.DataFrame) -> None:
    """Analyse salary distributions across demographics and departments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Salary Equity Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Salary distribution by department
    dept_order = df.groupby("department")["salary"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="department", y="salary", order=dept_order,
                palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("Salary by Department", fontsize=14)
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Salary by gender
    sns.boxplot(data=df, x="gender", y="salary", palette="Set2", ax=axes[0, 1])
    axes[0, 1].set_title("Salary by Gender", fontsize=14)

    # 3. Salary vs tenure scatter
    sns.scatterplot(data=df, x="tenure_years", y="salary", hue="department",
                    alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title("Salary vs Tenure", fontsize=14)
    axes[1, 0].legend(fontsize=7, loc="upper left")

    # 4. Salary by education
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    sns.violinplot(data=df, x="education", y="salary", order=edu_order,
                   palette="muted", inner="quartile", ax=axes[1, 1])
    axes[1, 1].set_title("Salary by Education Level", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/salary_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] salary_equity.png")


def statistical_tests(df: pd.DataFrame) -> None:
    """Run hypothesis tests and print results."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # ---- Chi-Square: Department vs Attrition ----
    contingency = pd.crosstab(df["department"], df["attrition"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n1. Chi-Square Test — Department vs Attrition")
    print(f"   Chi2 = {chi2:.4f}, df = {dof}, p-value = {p_chi:.6f}")
    print(f"   {'Significant' if p_chi < 0.05 else 'Not significant'} at alpha = 0.05")

    # ---- Chi-Square: Gender vs Attrition ----
    contingency_g = pd.crosstab(df["gender"], df["attrition"])
    chi2_g, p_chi_g, dof_g, _ = stats.chi2_contingency(contingency_g)
    print(f"\n2. Chi-Square Test — Gender vs Attrition")
    print(f"   Chi2 = {chi2_g:.4f}, df = {dof_g}, p-value = {p_chi_g:.6f}")
    print(f"   {'Significant' if p_chi_g < 0.05 else 'Not significant'} at alpha = 0.05")

    # ---- Independent t-test: Salary — Stayed vs Left ----
    stayed_salary = df.loc[df["attrition"] == 0, "salary"]
    left_salary = df.loc[df["attrition"] == 1, "salary"]
    t_stat, p_t = stats.ttest_ind(stayed_salary, left_salary, equal_var=False)
    print(f"\n3. Welch's t-Test — Salary (Stayed vs Left)")
    print(f"   Mean Stayed = ${stayed_salary.mean():,.0f}, Mean Left = ${left_salary.mean():,.0f}")
    print(f"   t = {t_stat:.4f}, p-value = {p_t:.6f}")
    print(f"   {'Significant' if p_t < 0.05 else 'Not significant'} at alpha = 0.05")

    # ---- t-test: Satisfaction — Stayed vs Left ----
    stayed_sat = df.loc[df["attrition"] == 0, "satisfaction"]
    left_sat = df.loc[df["attrition"] == 1, "satisfaction"]
    t_sat, p_sat = stats.ttest_ind(stayed_sat, left_sat, equal_var=False)
    print(f"\n4. Welch's t-Test — Satisfaction (Stayed vs Left)")
    print(f"   Mean Stayed = {stayed_sat.mean():.2f}, Mean Left = {left_sat.mean():.2f}")
    print(f"   t = {t_sat:.4f}, p-value = {p_sat:.6f}")
    print(f"   {'Significant' if p_sat < 0.05 else 'Not significant'} at alpha = 0.05")

    # ---- ANOVA: Salary across departments ----
    groups = [grp["salary"].values for _, grp in df.groupby("department")]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n5. One-Way ANOVA — Salary across Departments")
    print(f"   F = {f_stat:.4f}, p-value = {p_anova:.6f}")
    print(f"   {'Significant' if p_anova < 0.05 else 'Not significant'} at alpha = 0.05")

    # ---- Mann-Whitney U: Monthly Hours — Stayed vs Left ----
    stayed_hours = df.loc[df["attrition"] == 0, "monthly_hours"]
    left_hours = df.loc[df["attrition"] == 1, "monthly_hours"]
    u_stat, p_u = stats.mannwhitneyu(stayed_hours, left_hours, alternative="two-sided")
    print(f"\n6. Mann-Whitney U — Monthly Hours (Stayed vs Left)")
    print(f"   U = {u_stat:.4f}, p-value = {p_u:.6f}")
    print(f"   {'Significant' if p_u < 0.05 else 'Not significant'} at alpha = 0.05")

    print("=" * 70)


def correlation_analysis(df: pd.DataFrame) -> None:
    """Correlation heatmap of all numeric features."""
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["employee_id"])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] correlation_matrix.png")


def performance_analysis(df: pd.DataFrame) -> None:
    """Analyse performance scores across dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Performance Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Performance by department
    dept_perf = df.groupby("department")["performance"].mean().sort_values(ascending=False)
    sns.barplot(x=dept_perf.values, y=dept_perf.index, palette="Blues_d", ax=axes[0])
    axes[0].set_title("Mean Performance by Department", fontsize=14)
    axes[0].set_xlabel("Mean Performance Score")

    # 2. Performance vs Satisfaction scatter
    sns.scatterplot(data=df, x="satisfaction", y="performance",
                    hue=df["attrition"].map({0: "Stayed", 1: "Left"}),
                    alpha=0.5, ax=axes[1])
    axes[1].set_title("Performance vs Satisfaction", fontsize=14)

    # 3. Performance distribution by promotion
    sns.kdeplot(data=df[df["promoted_last_3yr"] == 1], x="performance",
                label="Promoted", fill=True, alpha=0.4, ax=axes[2])
    sns.kdeplot(data=df[df["promoted_last_3yr"] == 0], x="performance",
                label="Not Promoted", fill=True, alpha=0.4, ax=axes[2])
    axes[2].set_title("Performance by Promotion Status", fontsize=14)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/performance_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] performance_analysis.png")


def print_key_insights(df: pd.DataFrame) -> None:
    """Print key findings."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    attr_rate = df["attrition"].mean()
    print(f"  Overall attrition rate     : {attr_rate:.1%}")
    worst_dept = df.groupby("department")["attrition"].mean().idxmax()
    worst_rate = df.groupby("department")["attrition"].mean().max()
    print(f"  Highest attrition dept     : {worst_dept} ({worst_rate:.1%})")
    print(f"  Mean salary (overall)      : ${df['salary'].mean():,.0f}")
    print(f"  Mean satisfaction (stayed) : {df.loc[df['attrition']==0, 'satisfaction'].mean():.2f}")
    print(f"  Mean satisfaction (left)   : {df.loc[df['attrition']==1, 'satisfaction'].mean():.2f}")
    print(f"  Promotion rate             : {df['promoted_last_3yr'].mean():.1%}")
    print(f"  Avg monthly hours          : {df['monthly_hours'].mean():.0f}")
    print("=" * 70)


# ===========================================================================
# 3. Main
# ===========================================================================
def main() -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic HR dataset ...")
    df = generate_hr_data()
    df.to_csv(f"{OUTPUT_DIR}/hr_synthetic.csv", index=False)
    print(f"[Saved] hr_synthetic.csv  ({len(df)} rows)\n")

    basic_overview(df)
    attrition_analysis(df)
    salary_equity_analysis(df)
    statistical_tests(df)
    correlation_analysis(df)
    performance_analysis(df)
    print_key_insights(df)

    print("\nAll analyses complete. Charts saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
