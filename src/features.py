import pandas as pd

goemotions_df = pd.read_csv("../../personal-ai-agent-training/data/raw/goemotions.csv")


# -------------------------------------------------------------------
# Outlier detection (numerical data)
# -------------------------------------------------------------------

#no numerical continuous values other than time stamp
goemotions_df.select_dtypes(include="number").describe()




# -------------------------------------------------------------------
# Cleaning and preprocessing texts columns
# -------------------------------------------------------------------
# List of categorical columns (example: adjust as needed)
categorical_columns = ["subreddit", "author", "parent_id", "link_id"]