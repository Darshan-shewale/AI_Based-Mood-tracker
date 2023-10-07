import streamlit as st
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji

# Load the trained model and TF-IDF vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Emoji mapping
emotion_emojis = {
    'Sadness': ':cry:',
    'Joy': ':smile:',
    'Love': ':heart:',
    'Anger': ':angry:',
    'Fear': ':fearful:',
    'Unknown': ':question:',
}

# Define the emotion_labels dictionary to map text emotions to numerical labels
emotion_labels = {
    'Sadness': 0,
    'Joy': 1,
    'Love': 2,
    'Anger': 3,
    'Fear': 4,
}

# Streamlit UI
st.title('Emotion Prediction App')

comment = st.text_area('Enter your comment:')
if st.button('Predict Emotion'):
    if comment:
        # Preprocess the user's input comment
        def preprocess_comment(comment):
             if isinstance(comment, str):
                  comment = comment.lower()
                  comment = re.sub(r'http\S+', '', comment)
                  comment = re.sub(r'[^a-zA-Z\s]', '', comment)
                  tokens = word_tokenize(comment)
                  stop_words = set(stopwords.words('english'))
                  filtered_tokens = [word for word in tokens if word not in stop_words]
                  lemmatizer = WordNetLemmatizer()
                  lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
                  cleaned_comment = ' '.join(lemmatized_tokens)
             else:
                  cleaned_comment = ''
             return cleaned_comment

        cleaned_comment = preprocess_comment(comment)

        # Vectorize the cleaned comment
        input_features = tfidf_vectorizer.transform([cleaned_comment])

        # Predict the emotion
        emotion_id = model.predict(input_features)[0]

        # Map the numerical emotion prediction to text labels
        predicted_emotion = 'Unknown'
        for emotion, label in emotion_labels.items():
            if label == emotion_id:
                predicted_emotion = emotion

        # Get the corresponding emoji for the predicted emotion
        emoji_icon = emotion_emojis.get(predicted_emotion, ':question:')

        # Display the predicted emotion with the emoji
        st.write(f'Predicted Emotion: {predicted_emotion} {emoji.emojize(emoji_icon)}')

        # Suggest content based on the detected emotion (dummy content for demonstration)
        st.subheader('Suggested Content:')
        if predicted_emotion == 'Joy':
            st.write('1. Watch this funny video: [https://youtu.be/ZXcTvbLwsRs]')
            st.write('2. Read an uplifting article: [https://themindofsteel.com/best-motivation-ever/]')
        elif predicted_emotion == 'Sadness':
            st.write('1. Listen to calming music: [https://www.youtube.com/watch?v=lFcSrYw-ARY]')
            st.write('2. Read a heartwarming story: [https://youtu.be/Bi-7pho5XB8]')
        elif predicted_emotion == 'Anger':
            st.write('1. Try this anger management exercise: [https://www.youtube.com/watch?v=-Q20udWcaLg]')
            st.write('2. Read tips for managing anger: [https://www.youtube.com/watch?v=C1N4f1F0vDU]')
        else:
            st.write('No content suggestions available for this mood.')
    else:
        st.warning('Please enter a comment for prediction.')
