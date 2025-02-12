import pandas as pd
import torch
import re
import logging
import unicodedata
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Greek stopwords and lemmatizer
try:
    stop_words = set(stopwords.words('greek'))
except Exception as e:
    logging.error("Error loading Greek stopwords: %s", e)
    stop_words = set()

def remove_accents(text):
    """Removes Greek accents from characters."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def preprocess_text(text, remove_punctuation=True, remove_stopwords=True, remove_accents_flag=True):
    """Preprocesses Greek text by removing punctuation, stopwords, and applying lemmatization."""
    try:
        text = text.lower()
        if remove_accents_flag:
            text = remove_accents(text)
        text = re.sub(r'\d+', '', text)  # Remove numbers
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        logging.error("Error in text preprocessing: %s", e)
        return text

def load_and_preprocess_data(filepath):
    """Loads a CSV file, preprocesses the text, and encodes the categories."""
    try:
        df = pd.read_csv(filepath, delimiter='|')
        df = df[['article', 'category']].copy()
        label_encoder = LabelEncoder()
        df['category'] = label_encoder.fit_transform(df['category'])
        df['article'] = df['article'].apply(preprocess_text)
        return df, label_encoder
    except Exception as e:
        logging.error("Error loading and preprocessing data: %s", e)
        return None, None

def tokenize_function(examples):
    """Tokenizes text for input to the BERT model."""
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall, and F1-score for evaluation."""
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_and_evaluate_model(train_texts, train_labels, test_texts, test_labels, label_encoder):
    """Trains and evaluates the GreekBERT model."""
    try:
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        model = BertForSequenceClassification.from_pretrained(
            "nlpaueb/bert-base-greek-uncased-v1", num_labels=len(label_encoder.classes_))
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logging.info("Starting model training...")
        trainer.train()
        
        logging.info("Evaluating model...")
        results = trainer.evaluate()
        pd.DataFrame(results, index=[0]).to_csv("evaluation_results.csv", index=False)
        
        logging.info("Saving test predictions...")
        predictions = trainer.predict(test_dataset)
        test_texts_df = pd.DataFrame({"article": test_texts, "actual": test_labels, "predicted": predictions.predictions.argmax(axis=1)})
        test_texts_df.to_csv("test_predictions.csv", index=False)
        
        return results
    except Exception as e:
        logging.error("Error during training and evaluation: %s", e)
        return None

if __name__ == "__main__":
    logging.info("Initializing GreekBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    
    logging.info("Loading and preprocessing data...")
    df, label_encoder = load_and_preprocess_data("data_csv_file.csv")
    if df is not None:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['article'].tolist(), df['category'].tolist(), test_size=0.3, random_state=42
        )
        logging.info("Starting training and evaluation...")
        train_and_evaluate_model(train_texts, train_labels, test_texts, test_labels, label_encoder)
    else:
        logging.error("Data loading failed. Exiting program.")
