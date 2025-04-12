import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import ztest
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# PROJECT OBJECTIVE: 
# Analyze suicide death rates in the U.S. by gender, age, and year using visualizations and statistics.

# LOAD & CLEAN DATA
df = pd.read_csv(r"C:\Users\ANJALI DUBEY\Downloads\Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv")

# Check missing values
print("Missing values:\n", df.isnull().sum())
#CLEANING DATA
df["ESTIMATE"] = df["ESTIMATE"].fillna(df["ESTIMATE"].mode()[0])
df_clean = df[df["AGE"] != "All ages"]
df_clean["GENDER"] = df_clean["STUB_NAME"].str.extract(r'(Male|Female)', expand=False)
df_clean["AGE_NUM"] = pd.to_numeric(df_clean["AGE_NUM"], errors="coerce")
age_order = df_clean.drop_duplicates("AGE")[["AGE", "AGE_NUM"]].dropna().sort_values("AGE_NUM")["AGE"]



# ========== OBJECTIVE 1: Area map showing Average Suicide Rate Over Time ==========
plt.figure(figsize=(10, 5))
yearly_avg = df_clean.groupby("YEAR")["ESTIMATE"].mean()
plt.fill_between(yearly_avg.index, yearly_avg.values, color='lightcoral', alpha=0.7)
plt.plot(yearly_avg.index, yearly_avg.values, color='darkred', marker='*', linewidth=2)
plt.title("Average Suicide Rate Over the Years")
plt.xlabel("Year")
plt.ylabel("Avg Death Rate per 100,000")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()






# ========== OBJECTIVE 2: Age Group Comparison ==========
age_avg = df_clean.groupby("AGE")["ESTIMATE"].mean().sort_values(ascending=False)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor("#add8e6")
sns.barplot(x=age_avg.values, y=age_avg.index, palette="viridis", ax=ax)
ax.set_title("Average Suicide Rate by Age Group", fontsize=14, fontweight='bold')
ax.set_xlabel("Death Rate", fontsize=12)
ax.set_ylabel("Age Group", fontsize=12)
plt.tight_layout()
plt.show()




# ========== OBJECTIVE 3: Distribution & KDE ==========
sns.set_style("whitegrid") 
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor("#f3e6ff") 

sns.histplot(df_clean["ESTIMATE"], bins=30, kde=True, color="mediumorchid", edgecolor="black", ax=ax)
ax.set_title("Distribution of Suicide Death Rates", fontsize=14)
ax.set_xlabel("Death Rate per 100,000")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()




# ========== OBJECTIVE 4: Heatmap of Suicide Rates by Age & Year ==========
corr_matrix = df_clean.select_dtypes(include=['number']).corr()
plt.figure(figsize=(8, 5))
sns.set_style("darkgrid")
sns.heatmap(corr_matrix, annot=True, cmap="cividis", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features", fontsize=13)
plt.tight_layout()
plt.show()





# ========== OBJECTIVE 5: Age Group Comparison==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#e6f2ff') 
sns.boxplot(data=df_clean, x="AGE", y="ESTIMATE", palette="Paired")
plt.title("Distribution of Suicide Rates Across Age Groups", fontsize=14, fontweight='bold')
plt.xlabel("Age Group")
plt.ylabel("Death Rate per 100,000")
plt.xticks(rotation=45)
plt.grid(True, linestyle='-.', alpha=0.6,color="navy")
plt.tight_layout()
plt.show()




# ========== OBJECTIVE 6: Age Group Comparison as a Doughnut Chart ==========
top5_age_avg = age_avg.head(5)
colors = sns.color_palette("inferno", len(top5_age_avg))
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    top5_age_avg,
    labels=top5_age_avg.index,
    autopct='%1.1f%%',
    colors=colors,
    wedgeprops={'width': 0.4},  
    pctdistance=0.75,           
    textprops={'color': 'white', 'fontsize': 11}
)
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
fig.gca().add_artist(centre_circle)
plt.title("Proportion of Suicide Rates Among Top 5 Age Groups", fontsize=14, fontweight='bold')
plt.legend(
    top5_age_avg.index,
    title="Age Groups",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)
plt.axis('equal') 
plt.tight_layout()
plt.show()




# ========== OBJECTIVE 7: Conditional Filtering & Statistical Comparison ==========
max_year = yearly_avg.idxmax()
print(f"\n Year with highest avg suicide rate: {max_year} ({yearly_avg[max_year]:.2f})")

max_age = age_avg.idxmax()
print(f"Age group with highest avg suicide rate: {max_age} ({age_avg[max_age]:.2f})")

gender_age_df = df_clean[df_clean["GENDER"].isin(["Male", "Female"])]

# Z-Test: Comparing suicide rates between males and females
male_rates = gender_age_df[gender_age_df["GENDER"] == "Male"]["ESTIMATE"]
female_rates = gender_age_df[gender_age_df["GENDER"] == "Female"]["ESTIMATE"]

gender_age_df = df_clean[df_clean["GENDER"].isin(["Male", "Female"])]

male_rates = gender_age_df[gender_age_df["GENDER"] == "Male"]["ESTIMATE"].dropna()
female_rates = gender_age_df[gender_age_df["GENDER"] == "Female"]["ESTIMATE"].dropna()
print(f"Male entries: {len(male_rates)}, Female entries: {len(female_rates)}")

if len(male_rates) > 0 and len(female_rates) > 0:
    z_stat, p_val = ztest(male_rates, female_rates)
    print(f"\nZ-statistic: {z_stat:.2f}, P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Significant difference in suicide rates between males and females.")
    else:
        print("No significant difference found.")
else:
    print("Not enough data to perform Z-test.")
