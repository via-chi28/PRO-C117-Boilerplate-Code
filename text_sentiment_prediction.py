import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


train_data = pd.read_excel("./static/assets/data_files/text-emotion-training-dataset.xlsx")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text_Emotion"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model_files/Text_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

emo_code_url = {
    "anger": [0, "./static/assets/emoticons/Anger.png"],
    "fear": [1,"./static/assets/emoticons/Fear.png" ],
    "joy": [2, "./static/assets/emoticons/Joy.png"],
    "love": [3, "./static/assets/emoticons/Love.png"],
    "sadness": [4, "./static/assets/emoticons/Sadness.png"],
    "surprise": [5, "./static/assets/emoticons/Surprise.png"]
    }

def predict(text):
    #Define variables to store the emotion and emoticons
   
    #Preprocess the text and predict emotion
   