import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.stdout = open("EDA_summary_output.txt", "w")
plt.style.use("seaborn-v0_8")

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
df = pd.read_csv("data/processed/cleaned_with_features.csv")
df.columns = df.columns.str.strip()

print("\n--- FIRST 10 ROWS ---")
print(df.head())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- SUMMARY (ALL NUMERIC COLUMNS) ---")
print(df.describe())

print("\n--- MISSING VALUES (TOP 10) ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\n--- VIRAL DISTRIBUTION (PROPORTION) ---")
print(df["viral"].value_counts(normalize=True))

# ----------------------------------------
# 1. TARGET DISTRIBUTION
# ----------------------------------------

# Raw shares
plt.figure(figsize=(8, 4))
sns.histplot(df["shares"], bins=60, kde=False)
plt.title("Distribution of Shares (Raw)")
plt.xlabel("Shares")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("\n--- SUMMARY STATS FOR SHARES (RAW) ---")
print(df["shares"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]))

# Log shares
plt.figure(figsize=(8, 4))
sns.histplot(df["shares_log"], bins=60, kde=True)
plt.title("Distribution of Shares (Log Scale)")
plt.xlabel("Log Shares")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

print("\n--- SUMMARY STATS FOR SHARES_LOG ---")
print(df["shares_log"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]))

# Log shares by viral label
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="shares_log", hue="viral", bins=40, kde=True)
plt.title("Shares Log Distribution: Viral vs Non Viral")
plt.tight_layout()
plt.show()

print("\n--- SHARES_LOG BY VIRAL LABEL ---")
print(df.groupby("viral")["shares_log"].describe(percentiles=[0.1, 0.5, 0.9]))

# ----------------------------------------
# 2. TOP CORRELATIONS
# ----------------------------------------
corr_series = df.corr(numeric_only=True)["shares_log"].sort_values(ascending=False)
print("\n--- TOP 15 CORRELATED FEATURES WITH shares_log ---")
print(corr_series.head(15))

top_corr = corr_series.abs().head(15).index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Top 15 Features)")
plt.tight_layout()
plt.show()

print("\n--- CORRELATION MATRIX FOR TOP 15 FEATURES ---")
print(df[top_corr].corr())

# ----------------------------------------
# 3. CATEGORY / CHANNEL EFFECTS
# ----------------------------------------
channel_cols = [c for c in df.columns if c.startswith("data_channel")]
if channel_cols:
    channel_means = {col: df[df[col] == 1]["shares_log"].mean() for col in channel_cols}
    channel_rates = {col: df[df[col] == 1]["viral"].mean() for col in channel_cols}

    # Average log shares by channel
    plt.figure(figsize=(10, 4))
    plt.bar(
        [c.replace("data_channel_is_", "") for c in channel_means.keys()],
        list(channel_means.values()),
        color="steelblue",
    )
    plt.title("Average Log Shares by Channel")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n--- AVERAGE SHARES_LOG BY CHANNEL ---")
    channel_mean_series = pd.Series(channel_means)
    channel_mean_series.index = channel_mean_series.index.str.replace(
        "data_channel_is_", ""
    )
    print(channel_mean_series.sort_values(ascending=False))

    # Viral rate by channel
    plt.figure(figsize=(10, 4))
    plt.bar(
        [c.replace("data_channel_is_", "") for c in channel_rates.keys()],
        list(channel_rates.values()),
        color="salmon",
    )
    plt.title("Viral Rate by Channel")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n--- VIRAL RATE BY CHANNEL ---")
    channel_rate_series = pd.Series(channel_rates)
    channel_rate_series.index = channel_rate_series.index.str.replace(
        "data_channel_is_", ""
    )
    print(channel_rate_series.sort_values(ascending=False))

# ----------------------------------------
# 4. WEEKDAY PATTERNS
# ----------------------------------------
weekday_order = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

plt.figure(figsize=(10, 4))
df.groupby("weekday")["shares_log"].mean().reindex(weekday_order).plot(
    kind="bar", color="purple"
)
plt.title("Average Log Shares by Weekday")
plt.tight_layout()
plt.show()

print("\n--- AVERAGE SHARES_LOG BY WEEKDAY ---")
print(df.groupby("weekday")["shares_log"].mean().reindex(weekday_order))

plt.figure(figsize=(10, 4))
df.groupby("weekday")["viral"].mean().reindex(weekday_order).plot(
    kind="bar", color="green"
)
plt.title("Viral Rate by Weekday")
plt.tight_layout()
plt.show()

print("\n--- VIRAL RATE BY WEEKDAY ---")
print(df.groupby("weekday")["viral"].mean().reindex(weekday_order))

# ----------------------------------------
# 5. ENGINEERED FEATURE RELATIONSHIPS
# ----------------------------------------

# Keyword density vs log shares
plt.figure(figsize=(6, 4))
sns.scatterplot(x="keyword_density", y="shares_log", data=df, alpha=0.3)
plt.title("Keyword Density vs Log Shares")
plt.tight_layout()
plt.show()

print("\n--- CORRELATION: KEYWORD_DENSITY AND SHARES_LOG ---")
print(df[["keyword_density", "shares_log"]].corr())

# Article word count vs log shares
plt.figure(figsize=(6, 4))
sns.scatterplot(x="article_word_count", y="shares_log", data=df, alpha=0.3)
plt.title("Article Word Count vs Log Shares")
plt.tight_layout()
plt.show()

print("\n--- CORRELATION: ARTICLE_WORD_COUNT AND SHARES_LOG ---")
print(df[["article_word_count", "shares_log"]].corr())

# Keyword count vs log shares
plt.figure(figsize=(6, 4))
sns.scatterplot(x="keyword_count", y="shares_log", data=df, alpha=0.3)
plt.title("Keyword Count vs Log Shares")
plt.tight_layout()
plt.show()

print("\n--- CORRELATION: KEYWORD_COUNT AND SHARES_LOG ---")
print(df[["keyword_count", "shares_log"]].corr())

print("\n--- EDA COMPLETE ---")