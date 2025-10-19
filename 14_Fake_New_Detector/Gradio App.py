import gradio as gr
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model
model = tf.keras.models.load_model("fake_news_model.keras")

# Preprocessing setup
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
vocabulary_size = 5000
max_length = 20


def preprocess_text(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    seq = [one_hot(review, vocabulary_size)]
    padded = pad_sequences(seq, maxlen=max_length, padding="pre")
    return padded


def gradio_predict(text):
    preprocessed = preprocess_text(text)
    prob = model.predict(preprocessed)[0][0]
    label = "Real" if prob > 0.5 else "Fake"
    return f"{label} (confidence={prob:.4f})"


app = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Enter News Headline or Text",
        lines=3,
        placeholder="e.g., Breaking: Scientists discover cure for aging!",
    ),
    outputs="text",
    title="Fake News Classifier",
    description="Detect whether a news headline or article is Fake or Real using an LSTM model.",
)

app.launch()
