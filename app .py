import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ---------- Load the LSTM model with exception handling ----------
try:
    model = load_model('model.h5')
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------- Load the tokenizer with exception handling ----------
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    st.success("Tokenizer loaded successfully")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function to predict the next word
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure sequence length matches

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)

    # Map predicted index to word
    for words, index in tokenizer.word_index.items():
        if index == predicted_index:
            return words
    return None

# Streamlit UI
st.title('NEXT WORDS PREDICTION WITH LSTM AND EARLY STOPPING')
input_text = st.text_input('ENTER THE SEQUENCES OF WORDS')

if st.button('Predict the next words'):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length
    next_word = predict_next_word(input_text, model, tokenizer, max_sequence_len)
    if next_word:
        st.write(f'Next Word: {next_word}')
    else:
        st.warning("No prediction could be made for this input.")
