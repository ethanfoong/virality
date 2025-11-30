import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/cleaned_with_features.csv')

print(df.info())

#viral and non-viral
print(df["viral"].value_counts(normalize=True))

df[" shares"].hist(bins=50)
plt.show()

df["shares_log"].hist(bins=50)
plt.show()

#top correlated features with shares_log
print(df.corr(numeric_only=True)["shares_log"].sort_values(ascending=False).head(10))