import pandas as pd
import os
import urllib.request
import io

def fetch_sts_b():
    print("Fetching STS-B (Semantic Textual Similarity) data...")
    # Using a stable raw URL from a GLUE mirror or similar if possible.
    # Often found in widely used repos. 
    # Using a known mirror or the original GLUE source if accessible.
    # URL: https://dl.fbaipublicfiles.com/glue/data/STS-B.zip is the official zip.
    # Reading ZIP in memory to extract train.tsv without external tools?
    # Or just use a raw tsv from a github mirror.
    
    # Try another mirror or just use a small subset manually if needed.
    # HuggingFace datasets often host raw files.
    # Trying a different github raw url.
    # https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/glue_data/STS-B/train.tsv
    
    # Fallback to a hardcoded small sample if download fails, just to have "real" data structure.
    # Or try the official zip but that requires 'zipfile'.
    # Let's try fetching a different dataset that is definitely raw.
    # SST-2 train.tsv?
    # https://raw.githubusercontent.com/prrao87/fine-grained-sentiment-classification/master/data/sst-2/train.tsv
    # Let's try creating a small manual STS-B like file for the sake of the experiment if download keeps failing.
    # But user asked for "more real data from the internet".
    # I will try to download the 'SentEval' STS data.
    
    url = "https://raw.githubusercontent.com/PrincetonML/SISE-SentEval/master/data/downstream/STS/STS12-en-test/STS.input.MSRpar.txt"
    label_url = "https://raw.githubusercontent.com/PrincetonML/SISE-SentEval/master/data/downstream/STS/STS12-en-test/STS.gs.MSRpar.txt"
    
    print(f"Downloading from {url}")
    
    try:
        import ssl
        if hasattr(ssl, '_create_unverified_context'):
            context = ssl._create_unverified_context()
        else:
            context = None

        with urllib.request.urlopen(url, context=context) as r:
            text_data = r.read().decode('utf-8').strip().split('\n')
            
        with urllib.request.urlopen(label_url, context=context) as r:
            label_data = r.read().decode('utf-8').strip().split('\n')
            
        # Parse
        data = []
        for t, l in zip(text_data, label_data):
            # t is tab separated sentences
            parts = t.split('\t')
            if len(parts) >= 2:
                text = parts[0] + " " + parts[1]
                score = float(l)
                data.append({'text': text, 'label': score})
                
        df = pd.DataFrame(data)
        os.makedirs('data', exist_ok=True)
        out_path = 'data/stsb_sample.csv'
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} samples to {out_path}")
        
    except Exception as e:
        print(f"Failed to download: {e}")
        
    except Exception as e:
        print(f"Failed to download STS-B: {e}")

if __name__ == "__main__":
    fetch_sts_b()

