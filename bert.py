import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Load Greek stopwords
stop_words = set(stopwords.words('greek'))
stemmer = SnowballStemmer("greek")

def preprocess_text(text, remove_punctuation=True, remove_stopwords=True, use_stemmer=False, use_lemmatizer=False):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    if remove_punctuation:
        text = re.sub(r'[\W]+', ' ', text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if use_stemmer:
        tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['article', 'category']].copy()
    label_encoder = LabelEncoder()
    df['category'] = label_encoder.fit_transform(df['category'])
    df['article'] = df['article'].apply(preprocess_text)
    return df, label_encoder

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_and_evaluate_model(train_texts, train_labels, test_texts, test_labels, label_encoder):
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
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    print(results)
    pd.DataFrame(results, index=[0]).to_csv("evaluation_results.csv", index=False)
    
    predictions = trainer.predict(test_dataset)
    test_texts_df = pd.DataFrame({"article": test_texts, "actual": test_labels, "predicted": predictions.predictions.argmax(axis=1)})
    test_texts_df.to_csv("test_predictions.csv", index=False)
    
    return results


tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
df, label_encoder = load_and_preprocess_data("your_dataset.csv")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['article'].tolist(), df['category'].tolist(), test_size=0.3, random_state=42
)
train_and_evaluate_model(train_texts, train_labels, test_texts, test_labels, label_encoder)
