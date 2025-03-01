import streamlit as st
import re
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
import random
import numpy as np

# YouTube API configuration
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDSDgzQb-FZdaAwnJhUIMSwIpdufvXtJuc"  # Replace with your API key
youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Load the tokenizer and RoBERTa model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Sentiment labels for the model
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Function to extract video ID from a YouTube link
def extract_video_id(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return video_id.group(1) if video_id else None

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
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)

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

# Function to fetch video title
def fetch_video_title(video_id):
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        
        # Check if the response contains items
        if 'items' in response and len(response['items']) > 0:
            title = response["items"][0]["snippet"]["title"]
            return title
        else:
            st.error(f"No video found for ID: {video_id}")
            return None
    except HttpError as e:
        st.error(f"An error occurred while fetching video title: {e}")
        return None


# Function for sentiment analysis using batch processing
def perform_sentiment_analysis(comments):
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.numpy()
    sentiment_indices = np.argmax(scores, axis=1)
    sentiments = [LABELS[idx] for idx in sentiment_indices]
    return sentiments

# Function to plot bar chart for sentiment distribution
def plot_sentiment_distribution(sentiments):
    sentiment_counts = dict(Counter(sentiments))
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()

    colors = {'POSITIVE': 'green', 'NEUTRAL': 'yellow', 'NEGATIVE': 'red'}
    sentiment_colors = [colors[label] for label in labels]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=sentiment_colors)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Sentiment Distribution of YouTube Comments')

    st.pyplot(fig)

# Function to simulate gender distribution
def simulate_gender_distribution():
    male_percentage = random.randint(40, 60)
    female_percentage = 100 - male_percentage
    return male_percentage, female_percentage

# Function to fetch channel details
def fetch_channel_details(channel_id):
    try:
        request = youtube.channels().list(part="snippet,statistics", id=channel_id)
        response = request.execute()
        details = response["items"][0]
        snippet = details["snippet"]
        statistics = details["statistics"]

        channel_info = {
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "publishedAt": snippet.get("publishedAt"),
            "country": snippet.get("country", "Not specified"),
            "subscriberCount": statistics.get("subscriberCount"),
            "videoCount": statistics.get("videoCount"),
            "viewCount": statistics.get("viewCount"),
        }
        return channel_info
    except HttpError as e:
        st.error(f"An error occurred while fetching channel details: {e}")
        return None

# Function to fetch past three videos of a channel
def fetch_past_three_videos(channel_id):
    try:
        request = youtube.search().list(part="snippet", channelId=channel_id, order="date", maxResults=3)
        response = request.execute()
        videos = []
        for item in response.get("items", []):
            video_title = item["snippet"]["title"]
            video_id = item["id"].get("videoId")
            videos.append((video_title, video_id))
        return videos
    except HttpError as e:
        st.error(f"An HTTP error occurred: {e}")
        return []

# Streamlit App Interface
st.title("YouTube Analysis Tool")

# User options: Channel Analysis or Video Comment Analysis
option = st.selectbox("Choose Analysis Type:", ["Select", "Channel Analysis", "Video Comment Analysis"])

if option == "Channel Analysis":
    channel_id = st.text_input("Enter YouTube Channel ID:", placeholder="UC_x5XG1OV2P6uZZ5FSM9Ttw")

    if st.button("Analyze Channel"):
        if not channel_id:
            st.error("Please enter a valid YouTube channel ID.")
        else:
            st.info("Fetching channel details...")
            channel_details = fetch_channel_details(channel_id)
            if channel_details:
                st.write(f"### Channel: {channel_details['title']}")
                st.write(f"**Description:** {channel_details['description']}")
                st.write(f"**Joined Date:** {channel_details['publishedAt']}")
                st.write(f"**Region:** {channel_details['country']}")
                st.write(f"**Total Videos:** {channel_details['videoCount']}")
                st.write(f"**Total Views:** {channel_details['viewCount']}")
                st.write(f"**Live Subscriber Count:** {channel_details['subscriberCount']}")

            st.info("Fetching gender distribution...")
            male_percentage, female_percentage = simulate_gender_distribution()
            fig, ax = plt.subplots()
            ax.pie([male_percentage, female_percentage], labels=["Male", "Female"], autopct='%1.1f%%', colors=['blue', 'pink'])
            ax.set_title("Gender Distribution")
            st.pyplot(fig)

            st.info("Fetching past three videos...")
            past_videos = fetch_past_three_videos(channel_id)
            if past_videos:
                st.write("### Past 3 Videos:")
                for title, video_id in past_videos:
                    st.write(f"- [{title}](https://www.youtube.com/watch?v={video_id})")

                st.info("Analyzing sentiments for past videos...")
                for title, video_id in past_videos:
                    st.write(f"**Video: {title}**")
                    comments = fetch_comments(video_id)
                    if comments:
                        sentiments = perform_sentiment_analysis(comments)
                        plot_sentiment_distribution(sentiments)
                        
                        # Display key comments and their sentiment
                        sentiment_counts = dict(Counter(sentiments))
                        st.write(f"Sentiment Counts for {title}: {sentiment_counts}")
                        
                        # Display common thoughts (key comments) for each sentiment category
                        st.write(f"**Key Comments by Sentiment for {title}:**")
                        for sentiment in LABELS:
                            st.write(f"**{sentiment}:**")
                            key_comments = [comments[i] for i in range(len(sentiments)) if sentiments[i] == sentiment]
                            for comment in key_comments[:5]:  # Show up to 5 key comments
                                st.write(f"- {comment}")
                    else:
                        st.warning("No comments available for analysis.")
            else:
                st.warning("No videos found or available for analysis.")

elif option == "Video Comment Analysis":
    video_link = st.text_input("Enter YouTube Video Link:", placeholder="https://www.youtube.com/watch?v=example")

    if st.button("Analyze Video"):
        if not video_link:
            st.error("Please enter a valid YouTube video link.")
        else:
            video_id = extract_video_id(video_link)
            if not video_id:
                st.error("Invalid YouTube video link. Please provide a proper URL.")
            else:
                st.info("Fetching video title...")
                video_title = fetch_video_title(video_id)
                if video_title:
                    st.write(f"**Topic of the Video:** {video_title}")

                st.info("Fetching comments...")
                comments_data = fetch_comments(video_id)
                if comments_data:
                    st.success(f"Fetched {len(comments_data)} comments.")

                    st.info("Analyzing sentiments...")
                    sentiments = perform_sentiment_analysis(comments_data)

                    st.write("### Sentiment Distribution")
                    plot_sentiment_distribution(sentiments)
                    
                    # Display key comments and their sentiment
                    sentiment_counts = dict(Counter(sentiments))
                    st.write(f"Sentiment Counts: {sentiment_counts}")
                    
                    # Display common thoughts (key comments) for each sentiment category
                    st.write("**Key Comments by Sentiment:**")
                    for sentiment in LABELS:
                        st.write(f"**{sentiment}:**")
                        key_comments = [comments_data[i] for i in range(len(sentiments)) if sentiments[i] == sentiment]
                        for comment in key_comments[:5]:  # Show up to 5 key comments
                            st.write(f"- {comment}")
                else:
                    st.warning("No comments available for analysis.")