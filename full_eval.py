import pandas as pd
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from difflib import SequenceMatcher
import unicodedata as ud
from sklearn.preprocessing import LabelEncoder

def compute_similarity(text1, text2):
        """Computes similarity ratio between two texts using SequenceMatcher."""
        c_text1 = clean_text(text1)
        c_text2 = clean_text(text2)
        return SequenceMatcher(None, c_text1, c_text2).ratio()

def update_pred_with_similarity(label_list, pred_list, threshold=0.8):
    updated_pred_list = pred_list[:]  # Copy pred_list to avoid modifying the original list
    
    for i, pred in enumerate(pred_list):
        best_match = None
        best_ratio = 0

        for label in label_list:
            similarity = SequenceMatcher(None, label, pred).ratio()
            if similarity > best_ratio and similarity >= threshold:
                best_match = label
                best_ratio = similarity
        
        if best_match:
            updated_pred_list[i] = best_match  # Replace with the best match from label_list

    return updated_pred_list

def evaluate_model(df, similarity_threshold=0.8, output_file="llm_evaluation_results.csv"):
    """
    Evaluates the performance of an LLM's classification using similarity-based comparison.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'title', 'article', 'category', and LLM-generated responses.
        similarity_threshold (float): Threshold for similarity score to determine correct classification. Default is 0.9.
        output_file (str): Path to save the evaluation results CSV file. Default is 'evaluation_results.csv'.
    """
    df.iloc[:,3] = df.iloc[:, 3].apply(clean_text)
    df.iloc[:,2] = df.iloc[:, 2].apply(clean_text)
    true_categories = df.iloc[:, 2].astype(str).tolist()
    # print(predicted_categories)
    llm_column_name = df.columns[3]  # Automatically detect the LLM response column (4th column)
    llm_responses = df[llm_column_name].astype(str).to_list()
    print(list(set(true_categories)))
    predicted_categories = update_pred_with_similarity(list(set(true_categories)),llm_responses,threshold=similarity_threshold)
    print(predicted_categories)

    # Compute similarity scores
    # similarity_scores = [compute_similarity(true_cat_str, pred) for true_cat_str, pred in zip(true_categories, llm_responses)]
    # Determine predictions based on similarity threshold
    # predicted_categories = [true_cat_str if score >= similarity_threshold else "incorrect" for true_cat_str, score in zip(true_categories, similarity_scores)]
    # print(predicted_categories)
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_categories, predicted_categories)
    precision, recall, f1, _ = precision_recall_fscore_support(true_categories, predicted_categories, average="macro", zero_division=0)
    eval_loss = 1 - accuracy  # Loss defined as 1 - accuracy
    
    # Create results DataFrame
    evaluation_results = pd.DataFrame([[llm_column_name, eval_loss, accuracy, precision, recall, f1]],
                                      columns=["model", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1"])
    
    # Append results to CSV file
    try:
        existing_results = pd.read_csv(output_file)
        evaluation_results = pd.concat([existing_results, evaluation_results], ignore_index=True)
    except FileNotFoundError:
        pass
    
    evaluation_results.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")

def clean_text(text):
    """
    Trims leading and trailing whitespaces, removes special characters, and normalizes the text.
    """
    json_value = json_get_value(text,"news_article_type")
    if json_value:
        text = json_value

    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    removed_accents = ud.normalize('NFD',text.strip().lower()).translate(d)
    
    removed_spaces_sp_accents = ''.join(e for e in removed_accents if e.isalpha() or e ==' ')
    
    return removed_spaces_sp_accents


# Example usage:
# df = pd.read_csv("your_data.csv")
# evaluate_model(df, "llm_response_column")
# Example usage:
# df = pd.read_csv("your_data.csv")
# evaluate_model(df, "llm_response_column")



# Extract JSON data from the fourth column
def json_get_value(json_string,typename="news_article_type"):
    if not json_string:
        return json_string
    try:
        parsed_data = json.loads(json_string)  # Convert JSON string to dictionary
        value = parsed_data[typename]  # Get the value associated with typename
        return value
    except ValueError:
        return False
        
def read_csv_and_extract_json(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path,encoding='utf-8',delimiter='|')
    # Ensure the fourth column exists
    if df.shape[1] < 4:
        raise ValueError("CSV does not have at least four columns.")

    df.iloc[:,3] = df.iloc[:, 3].apply(json_get_value)
    
    return df

df = read_csv_and_extract_json("./output_llama_3_1_8b.csv")


# print(df)
# df.to_csv("df.csv")
evaluate_model(df)