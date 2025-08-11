import pandas as pd


#-------------------------------------------------------------------
goemotions_df = pd.read_csv("../../personal-ai-agent-training/data/raw/goemotions.csv")

#-------------------------------------------------------------------
# emotion_categories greatest to least common
#-------------------------------------------------------------------
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
goemotions_df[emotion_columns].sum().sort_values(ascending = False)

# neutral is most common but grief is least common

