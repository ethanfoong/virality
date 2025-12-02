import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # Project paths relative to this file
    project_root = Path(__file__).resolve().parents[1]
    data_raw = project_root / "data" / "raw"
    data_processed = project_root / "data" / "processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    raw_path = data_raw / "OnlineNewsPopularity.csv"

    print(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # 1. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows")

    # 2. Check missing values
    missing_total = df.isna().sum().sum()
    print(f"Total missing values: {missing_total}")

    # 3. Target transforms

    # shares_log = log1p(shares)
    df["shares_log"] = np.log1p(df[" shares"])

    # Viral label: top 10 percent of shares
    viral_threshold = df[" shares"].quantile(0.90)
    print(f"Viral threshold (90th percentile of shares): {viral_threshold:.1f}")
    df["viral"] = (df[" shares"] >= viral_threshold).astype(int)

    # Save cleaned base
    cleaned_base_path = data_processed / "cleaned_base.csv"
    df.to_csv(cleaned_base_path, index=False)
    print(f"Wrote: {cleaned_base_path}")

    # 4. Simple engineered features

    # Headline length in words
    if "n_tokens_title" in df.columns:
        df["headline_word_count"] = df["n_tokens_title"]

    # Article length in words
    if "n_tokens_content" in df.columns:
        df["article_word_count"] = df["n_tokens_content"]

    # Number of keywords
    if "num_keywords" in df.columns:
        df["keyword_count"] = df["num_keywords"]

    # Keyword density: keywords per content word
    if "num_keywords" in df.columns and "n_tokens_content" in df.columns:
        denom = df["n_tokens_content"].replace({0: np.nan})
        df["keyword_density"] = df["num_keywords"] / denom

    # Weekday and weekend features
    weekday_cols = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
    ]
    existing_weekday_cols = [c for c in weekday_cols if c in df.columns]

    if existing_weekday_cols:
        # Categorical weekday label from one hot columns
        df["weekday"] = (
            df[existing_weekday_cols]
            .idxmax(axis=1)
            .str.replace("weekday_is_", "", regex=False)
        )

        # Weekend flag
        weekend_cols = [
            c
            for c in ["weekday_is_saturday", "weekday_is_sunday"]
            if c in df.columns
        ]
        if weekend_cols:
            df["is_weekend"] = df[weekend_cols].max(axis=1).astype(int)

    # Save enriched dataset
    cleaned_with_features_path = data_processed / "cleaned_with_features.csv"
    df.to_csv(cleaned_with_features_path, index=False)
    print(f"Wrote: {cleaned_with_features_path}")


if __name__ == "__main__":
    main()
