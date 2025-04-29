from textblob import TextBlob
from fastapi import HTTPException
import pandas as pd

# CSV paths
CSV_PATHS = {
    "general": "election_sentiment_data.csv",
    "modi": "modi.csv",
    "rahul": "rahulgandhi.csv"
}

# Sentiment analysis function
def get_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text using TextBlob.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Analyze a single tweet
def analyze_single_tweet(text: str) -> dict:
    sentiment = get_sentiment(text)
    return {"tweet": text, "sentiment": sentiment}

# Load and analyze any CSV file
def analyze_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Tweet" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'Tweet' column.")
    if "Predicted Sentiment" not in df.columns:
        df["Predicted Sentiment"] = df["Tweet"].apply(get_sentiment)
    return df

# Generate summary statistics
def get_summary_statistics(dataset: str = "general") -> dict:
    try:
        path = CSV_PATHS.get(dataset)
        if not path:
            raise ValueError("Invalid dataset requested.")

        df = analyze_csv(path)
        sentiment_counts = df["Predicted Sentiment"].value_counts().to_dict()
        return {
            "summary": sentiment_counts,
            "total": len(df),
            "preview": df[["Tweet", "Predicted Sentiment"]].head(10).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
