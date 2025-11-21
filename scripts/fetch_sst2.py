import pandas as pd
import urllib.request
import io
import ssl
import os

# Disable SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def fetch_sst2():
    # Try a different stable source: TREC-6 from cognitive computation
    # Not SST-2 but a good text classification benchmark.
    # URL: https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label
    # Format: Label text...
    # We can parse this.
    
    url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    print(f"Downloading TREC-6 from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            # Latin-1 encoding often used in older datasets
            content = response.read().decode('latin-1')
            
        lines = content.strip().split('\n')
        data = []
        for line in lines:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                label = parts[0]
                text = parts[1]
                # Map categorical labels to integers for regression/classification
                # ABBR, DESC, ENTY, HUM, LOC, NUM
                data.append({'text': text, 'label_str': label})
                
        df = pd.DataFrame(data)
        
        # Map labels to numeric
        label_map = {l: i for i, l in enumerate(df['label_str'].unique())}
        df['label'] = df['label_str'].map(label_map).astype(float)
        
        # Save
        os.makedirs('data', exist_ok=True)
        out_path = 'data/trec_train.csv'
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} samples to {out_path}")
        print(df.head())
        
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    fetch_sst2()

