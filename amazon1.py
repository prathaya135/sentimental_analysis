# %%
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag
import nltk
import pandas_profiling as pp
# from nltk. import 
# nltk.download()

# %%
df = pd.read_csv('Review.csv')
print(df.shape)
df = df.head(500)



# %%
# model1
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

# %%
sia.polarity_scores('I am so happy!')

# %%
sia.polarity_scores('This is the worst thing ever.')



# %%
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']   #comment
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
# res

# %%
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# %%
# model 2 roberta
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# %%
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)



# %%
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# %%
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

# %%
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# %%
#  pipeline
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

# %%
sent_pipeline('I love sentiment analysis!')
count_positive=0
count_negative=0


file1 = open('text.txt','w')
file2 = open('positive.txt','w')
count1=0
count2=0
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        result = sent_pipeline(text)
        t= result[0]
        if t['label']=="POSITIVE":
            count_positive=count_positive+1
            if count2<6:
                file2.write(str(count2)+".")
                file2.write(text)
                file2.write("\n")
                count2=count2+1
        if t['label']=="NEGATIVE":
            count_negative=count_negative+1
            if count1<6:
                file1.write(str(count1)+".")
                file1.write(text)
                file1.write("\n")
                count1=count1+1
    except RuntimeError:
        print(f'Broke for id {myid}')

file1.close()
file2.close()

print(count_positive)
print(count_negative)

file = open('sample.txt','w')
count_negative= str(count_negative)
file.write(count_negative)
file.write("\n")
count_positive= str(count_positive)
file.write(count_positive)
file.close()





