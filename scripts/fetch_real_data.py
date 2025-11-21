import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os
import ssl

# Disable SSL verification to avoid certificate errors in some envs
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def fetch_and_save_data():
    print("Fetching 20 newsgroups data...")
    # Fetch a subset to be fast
    categories = ['sci.space', 'talk.politics.misc']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    texts = newsgroups.data
    # 0 for sci.space, 1 for politics
    targets = newsgroups.target.astype(float) 
    
    # Create DataFrame
    df = pd.DataFrame({'text': texts, 'label': targets})
    
    # Filter empty strings
    df = df[df['text'].str.strip().str.len() > 0]
    
    # Save
    os.makedirs('data', exist_ok=True)
    out_path = 'data/20news_binary.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")

if __name__ == "__main__":
    fetch_and_save_data()

