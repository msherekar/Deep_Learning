import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def download_kaggle_competition_data(competition_name, kaggle_json_path):
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition_name, path="./data", unzip=True)

def load_data(train_path, test_path):
    tYX = pd.read_csv(train_path).set_index('id')
    vX = pd.read_csv(test_path).set_index('id')
    return tYX, vX

def kmer_transform(seq, size=5):
    return ' '.join(''.join(kmer) for kmer in zip(*[seq[i:] for i in range(size)])).lower()

def preprocess_data(df):
    df['words'] = df['DNA'].map(kmer_transform)
    return df

def train_model(X_train, X_test, y_train, y_test):
    model = LinearSVC(random_state=0, max_iter=10000)
    model.fit(X_train, y_train)
    return model

def main():
    logger = setup_logging()
    kaggle_competition = "4mar24jh-genomics"
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if not os.path.exists("./data"): os.makedirs("./data")
    download_kaggle_competition_data(kaggle_competition, kaggle_json_path)
    
    train_file = "./data/trainYX_genomics.csv"
    test_file = "./data/testX_genomics.csv"
    tYX, vX = load_data(train_file, test_file)
    
    tYX = preprocess_data(tYX)
    vX = preprocess_data(vX)
    
    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(tYX['words'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, tYX['Y'], test_size=0.2, stratify=tYX['Y'], random_state=42)
    model = train_model(X_train, X_test, y_train, y_test)
    
    logger.info(f"Model Training Complete. Train Accuracy: {model.score(X_train, y_train):.3f}, Test Accuracy: {model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    main()
