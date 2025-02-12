import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from difflib import SequenceMatcher
import unicodedata as ud

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

def evaluate_model(df, similarity_threshold=0.8, average = "weighted", output_file="llm_evaluation_results.csv"):
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
    # print(list(set(true_categories)))
    predicted_categories = update_pred_with_similarity(list(set(true_categories)),llm_responses,threshold=similarity_threshold)
    # print(predicted_categories)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_categories, predicted_categories)
    precision, recall, f1, _ = precision_recall_fscore_support(true_categories, predicted_categories, average=average, zero_division=0)
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
        print("here")
        return ''
    try:
        parsed_data = json.loads(json_string)  # Convert JSON string to dictionary
        value = parsed_data[typename]  # Get the value associated with typename
        return value
    except ValueError:
        return False
        
def llm_result_json_to_df(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path,encoding='utf-8',delimiter='|')
    # Ensure the fourth column exists
    df = df.replace({np.nan: None})
    if df.shape[1] < 4:
        print(csv_path)
        raise ValueError("CSV does not have at least four columns.")

    df.iloc[:,3] = df.iloc[:, 3].apply(json_get_value)
    
    return df




def evaluate_folder_recursive(folder_path, similarity_threshold=0.8, average = "weighted", eval_filename="llm_evaluation_results.csv"):
    """
    Recursively evaluate all result CSV files in a folder and its subfolders
    using the evaluate_model function. Ignores non-CSV files.

    Args:
        folder_path (str): The path to the folder containing CSV files.
    """

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            eval_path = os.path.join(root,eval_filename)
            if os.path.exists(eval_path):
                os.remove(eval_path)
            if file_name.endswith("b.csv"):
                file_path = os.path.join(root, file_name)
                dataframe = llm_result_json_to_df(str(file_path))
                evaluate_model(dataframe,similarity_threshold=similarity_threshold, average=average, output_file=str(eval_path))

evaluate_folder_recursive("results_16k/",eval_filename="llm_evaluation_results_weighted.csv")
# print(df)
# df.to_csv("df.csv")
# evaluate_model(df)