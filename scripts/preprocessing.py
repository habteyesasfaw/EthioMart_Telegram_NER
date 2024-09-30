import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


nltk.download('punkt')

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def preprocess_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    normalized_tokens = [token.replace('አ', 'እ') for token in tokens]
    return ' '.join(normalized_tokens)

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if 'Message' not in df.columns:
        print("Error: 'Message' column not found in DataFrame.")
        return df
    
    # Apply the preprocessing function to the 'Message' column
    df['cleaned_message'] = df['Message'].apply(preprocess_text)
    return df

def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    try:
        df.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")

def run_preprocessing(input_file: str, output_file: str):
    raw_data = load_data(input_file)
    if raw_data.empty:
        print("No data to process. Exiting.")
        return
    cleaned_data = preprocess_dataset(raw_data)
    save_preprocessed_data(cleaned_data, output_file)

if __name__ == "__main__":
    input_file = 'telegram_data.csv'
    output_file = 'preprocessed_telegram_data.csv'
    run_preprocessing(input_file, output_file)
