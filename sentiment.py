from textblob import TextBlob
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd

CSV_Path = "election_sentiment_data.csv"
def get_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text using TextBlob.
    Returns 'positive', 'negative', or 'neutral' based on the polarity score.
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

# single tweet sentiment analysis
def analyze_single_tweet(text: str) -> dict:
    """
    Analyze the sentiment of a single tweet.
    Returns a dictionary with the tweet and its sentiment.
    """
    sentiment = get_sentiment(text)
    return {"tweet": text, "sentiment": sentiment}
    
# analyze total dataset sentiment
def analyze_csv_from_filesystem(path: str = CSV_Path) -> dict:
    df = pd.read_csv(path)
    if "Tweet" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'Tweet' column.")
    df["Predicted Sentiment"] = df["Tweet"].apply(get_sentiment)
    return df
# get summary statstistics
def get_summary_statistics(path: str = CSV_Path) -> dict:
    try:
        df = pd.read_csv(path)
        
        # Add sentiment if not present
        if "Predicted Sentiment" not in df.columns:
            df["Predicted Sentiment"] = df["Tweet"].apply(get_sentiment)
        
        sentiment_counts = df["Predicted Sentiment"].value_counts().to_dict()
        return {
            "summary": sentiment_counts,
            "total": len(df),
            "preview": df[["Tweet", "Predicted Sentiment"]].head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}