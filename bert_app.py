import streamlit as st
import pandas as pd
import numpy as np

# Core packages for text processing.
import string
import re
import warnings
import time
import datetime

# for building our model
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel

# Setting some options for general use.
import os
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# -----------------------------------------------------------------

st.markdown("""
<style>
div.stButton > button:first-child {
background-color: #00cc00;
color:black;
font-size:15px;
height:2.7em;
width:20em;
border-radius:10px 10px 10px 10px;}
</style>
    """,
            unsafe_allow_html=True
            )

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTI8DRxgEP4PMaChfWJQKulfwMWdF486bB0SF0ZHXkgS5z4gc2Jd7EGKC8-gjjKWNxEUlQ&usqp=CAU")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# -------------------------------------------------------------


os.environ["WANDB_API_KEY"] = "0"  ## to silence warning

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()  # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

# Bert Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

max_length = 128
batch_size = 128


def bert_encode(data):
    tokens = tokenizer.batch_encode_plus(data, max_length=max_length, padding='max_length', truncation=True)

    return tf.constant(tokens['input_ids'])


def bert_tweets_model():
    bert_encoder = TFBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    last_hidden_states = bert_encoder(input_word_ids)[0]
    x = tf.keras.layers.SpatialDropout1D(0.2)(last_hidden_states)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = tf.keras.layers.Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(input_word_ids, outputs)

    return model


with strategy.scope():
    model = bert_tweets_model()
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    # model.summary()


model.load_weights(r"C:\Users\abayo\Desktop\2056338 DATA ANALYTICS MAJOR PROJECT AND PLACEMENT\2056338 TRAINED MODEL\sentiment_weights.h5")


def decode_sentiment(score):
    if score == 0:
        return "negative"
    elif score == 1:
        return "neutral"
    else:
        return "positive"


st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def predict(text):
    start_at = time.time()
    # Tokenize text
    x_encoded = bert_encode([text])
    # Predict
    score = model.predict([x_encoded])[0]
    # Decode sentiment
    label = decode_sentiment(np.argmax(score))

    return {"label": label, "score": score,
            "elapsed_time": time.time() - start_at}

st.image("https://s33842.pcdn.co/wp-content/uploads/2021/11/welcome-buddy-1.jpg", width=1000)
st.title("Sentiment Analysis for ARU Welcome Buddy Scheme")
text = st.text_input("Please enter your chat or review")

if (st.button('Submit')):
    l1 = predict(text)
    result = l1['label'].title()
    st.success(result)



