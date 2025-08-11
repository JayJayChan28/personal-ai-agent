import pandas as pd
pd.set_option('display.max_columns', None)

#-------------------------------------------------------------------
# Dataset: GoEmotions
#-------------------------------------------------------------------

goemotions_df = pd.read_pickle("../../personal-ai-agent-training/data/interim/goemotions_df.pkl")
#-------------------------------------------------------------------
# Check for missing values
#-------------------------------------------------------------------
goemotions_df.isnull().sum()
#no missing values in dataset

goemotions_df.info()



#-------------------------------------------------------------------
# schema enforcement
#-------------------------------------------------------------------
#checking duplicate rows
duplicates = goemotions_df.duplicated()
duplicates.sum() # no duplicate rows

#-------------------------------------------------------------------
# Converting dtype of emotion columns to boolean
#-------------------------------------------------------------------
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

goemotions_df[emotion_columns] = goemotions_df[emotion_columns].astype(bool)


#-------------------------------------------------------------------# Outlier detection (numerical data)
#unclear examples
#-------------------------------------------------------------------
#shape of the dataframe with unclear examples
goemotions_df[goemotions_df['example_very_unclear'] == True].shape

#removing the unclear examples
goemotions_df_cleaned = goemotions_df[goemotions_df['example_very_unclear'] != True]
goemotions_df_cleaned.to_pickle("../../personal-ai-agent-training/data/interim/goemotions_df_cleaned.pkl")
