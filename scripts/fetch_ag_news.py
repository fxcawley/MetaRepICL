import pandas as pd
import urllib.request
import io
import os
import ssl

def fetch_ag_news():
    # URL for AG News train.csv
    # Trying a stable source.
    # https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
    
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    print(f"Downloading AG News from {url}...")
    
    # Disable SSL verify
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
            
        # AG News CSV usually has no header: "Class Index", "Title", "Description"
        df = pd.read_csv(io.StringIO(content), header=None, names=["label", "title", "description"])
        
        # Concatenate Title and Description
        df['text'] = df['title'] + " " + df['description']
        
        # Labels are 1-4. Map to 0-3.
        df['label'] = df['label'].astype(float) - 1.0
        
        # Select columns
        df = df[['text', 'label']]
        
        # Take a random sample to keep it manageable
        df = df.sample(n=5000, random_state=42)
        
        os.makedirs('data', exist_ok=True)
        out_path = 'data/ag_news_train.csv'
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} samples to {out_path}")
        print(df.head())
        
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    fetch_ag_news()

