import streamlit as st
import re
import requests
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter

# YouTube API configuration
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDSDgzQb-FZdaAwnJhUIMSwIpdufvXtJuc"  # Replace with your API key

youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Load the tokenizer and DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Function to extract the video ID from a YouTube link
def extract_video_id(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return video_id.group(1) if video_id else None

# Function to fetch the video title
def fetch_video_title(video_id):
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        title = response["items"][0]["snippet"]["title"]
        return title
    except HttpError as e:
        st.error(f"An error occurred while fetching video title: {e}")
        return None

# Function to preprocess comments
def preprocess_comment(comment):
    comment = re.sub(r"http\S+|www\S+|https\S+", "", comment)  # Remove URLs
    comment = re.sub(r"[^A-Za-z0-9\s]+", "", comment)  # Remove non-alphanumeric characters
    comment = re.sub(r"\s+", " ", comment).strip()  # Remove extra spaces
    return comment

# Function to fetch comments from a YouTube video
def fetch_comments(video_id):
    try:
        comments_data = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        
        while request:
            response = request.execute()
            for item in response.get("items", []):
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comment = comment_snippet.get('textDisplay', '')
                comment = preprocess_comment(comment)
                comments_data.append(comment)
            request = youtube.commentThreads().list_next(request, response)
        
        return comments_data
    except HttpError as e:
        st.error(f"An HTTP error occurred: {e}")
        return []

# Function to perform sentiment analysis with 3 categories: Positive, Negative, and Neutral
def perform_sentiment_analysis(comments):
    sentiments = []
    for comment in comments:
        # Tokenize and encode the comment with truncation for efficiency
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        # Get predictions from the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Classify sentiment based on model's output
        sentiment_score = torch.argmax(outputs.logits, dim=1).item()
        if sentiment_score == 0:
            sentiment = 'NEGATIVE'
        elif sentiment_score == 1:
            sentiment = 'NEUTRAL'
        else:
            sentiment = 'POSITIVE'
        sentiments.append(sentiment)
    
    return sentiments

# Function to plot bar chart for sentiment distribution
def plot_sentiment_distribution(sentiments):
    sentiment_counts = dict(Counter(sentiments))
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    
    # Assign colors based on sentiment type
    colors = {'POSITIVE': 'green', 'NEUTRAL': 'yellow', 'NEGATIVE': 'red'}
    sentiment_colors = [colors[label] for label in labels]
    
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=sentiment_colors)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Sentiment Distribution of YouTube Comments')

    # Display the bar chart with Streamlit
    st.pyplot(fig)

# Streamlit App Interface
st.title("YouTube Comments Sentiment Analysis")

video_link = st.text_input("Enter YouTube Video Link:", placeholder="https://www.youtube.com/watch?v=example")

if st.button("Analyze"):
    if not video_link:
        st.error("Please enter a valid YouTube video link.")
    else:
        video_id = extract_video_id(video_link)
        if not video_id:
            st.error("Invalid YouTube video link. Please provide a proper URL.")
        else:
            # Fetch the video title dynamically
            st.info("Fetching video title...")
            video_title = fetch_video_title(video_id)
            if video_title:
                st.write(f"**Topic of the Video:** {video_title}")
            
            st.info("Fetching comments...")
            comments_data = fetch_comments(video_id)
            if comments_data:
                st.success(f"Fetched {len(comments_data)} comments.")
                
                # Perform sentiment analysis
                st.info("Analyzing sentiments...")
                sentiments = perform_sentiment_analysis(comments_data)
                
                # Show bar chart of sentiment distribution
                st.write("### Sentiment Distribution")
                plot_sentiment_distribution(sentiments)
                
                # Count sentiment distribution
                positive_count = sentiments.count("POSITIVE")
                neutral_count = sentiments.count("NEUTRAL")
                negative_count = sentiments.count("NEGATIVE")
                
                # Display sentiment distribution counts
                st.write(f"**Positive Comments:** {positive_count}")
                st.write(f"**Neutral Comments:** {neutral_count}")
                st.write(f"**Negative Comments:** {negative_count}")
                
                # Group comments by sentiment and show key comments
                st.write("### Key Comments Distribution by Sentiment:")

                st.write("**Positive Comments:**")
                positive_comments = [comments_data[i] for i in range(len(comments_data)) if sentiments[i] == "POSITIVE"]
                for comment in positive_comments[:5]:  # Show top 5 positive comments
                    st.write(f"- {comment}")

                st.write("**Neutral Comments:**")
                neutral_comments = [comments_data[i] for i in range(len(comments_data)) if sentiments[i] == "NEUTRAL"]
                for comment in neutral_comments[:5]:  # Show top 5 neutral comments
                    st.write(f"- {comment}")

                st.write("**Negative Comments:**")
                negative_comments = [comments_data[i] for i in range(len(comments_data)) if sentiments[i] == "NEGATIVE"]
                for comment in negative_comments[:5]:  # Show top 5 negative comments
                    st.write(f"- {comment}")

                # Common topic-related comments (display first few from each sentiment)
                st.write(f"### What People Think About the Topic: {video_title}")
                st.write("These comments reflect common thoughts about the video topic:")
                for comment in comments_data[:5]:  # Display top 5 comments
                    st.write(f"- {comment}")
            else:
                st.warning("No comments fetched or available for analysis.")
