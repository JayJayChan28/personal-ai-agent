import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


goemotions_df = pd.read_csv("../../personal-ai-agent-training/data/raw/goemotions.csv")

#-------------------------------------------------------------------
# proportion of emmotions by each catagory
#-------------------------------------------------------------------
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
emotions_prop = (goemotions_df[emotion_columns].sum()/len(goemotions_df)).sort_values(ascending=False)

plt.figure(figsize=(20, 10))
plt.xticks(rotation=45)
plt.bar(x=emotions_prop.index, height=emotions_prop.values)


# seems to be some class imbalance, def need to apply some smote techniques
# tune threshold
# consider ensemble methods to capture and learn patterns of lower frequency emotions


#-------------------------------------------------------------------
# emotion_categories greatest to least common
#-------------------------------------------------------------------