import pandas as pd
import re
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

goemotions_df = pd.read_csv("../../personal-ai-agent-training/data/raw/goemotions.csv")


# -------------------------------------------------------------------
# Outlier detection (numerical data)
# -------------------------------------------------------------------

#no numerical continuous values other than time stamp
goemotions_df.select_dtypes(include="number").describe()




# -------------------------------------------------------------------
# Cleaning and preprocessing texts columns
# -------------------------------------------------------------------


URL_RE  = re.compile(r'https?://\S+|www\.\S+')
USER_RE = re.compile(r'@\w+')
SUB_RE  = re.compile(r'\br/\w+')

def clean_text(text):
    text = URL_RE.sub(' <URL> ', text) #replaces URLs with <URL>
    text = USER_RE.sub(' <USER> ', text) #replaces user mentions with <USER>
    text = SUB_RE.sub(' <SUBREDDIT> ', text) #replaces subreddit mentions with <SUBREDDIT>
    text = text.replace('\n', ' ') #replaces newlines with space
    text = text.replace('\t', ' ') #replaces tabs with space
    text = text.strip() #removes leading and trailing spaces
    text = re.sub(r'\s+', ' ', text) #removes multiple spaces
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    text = ' '.join(tokens)
    return text

goemotions_df['text'] = goemotions_df['text'].apply(clean_text)


text = 'dog cat fish'
text.split()