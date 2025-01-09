import csv
from difflib import SequenceMatcher
import unicodedata as ud
import re

from tools import add_prefix_filename_from_path

def remove_prefix(text, prefix="Απαντηση:"):
    """
    Removes the prefix "Απαντηση: " from the given text if it exists.
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def clean_text(text):
    """
    Trims leading and trailing whitespaces, removes special characters, and normalizes the text.
    """
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    removed_accents = ud.normalize('NFD',text.strip().lower()).translate(d)
    
    print(removed_accents)
    removed_spaces_sp_accents = ''.join(e for e in removed_accents if e.isalpha() or e ==' ')
    
    return removed_spaces_sp_accents

def is_similar(word1, word2, threshold=0.7):
    """
    Checks if two words are similar with at least the given threshold similarity (default 70%).
    Cleans the input words before comparison.
    """
    word1_cleaned = clean_text(word1)
    word2_cleaned = clean_text(word2)
    similarity = SequenceMatcher(None, word1_cleaned, word2_cleaned).ratio()
    print(similarity)
    similarity_boolean = similarity >= threshold
    return similarity_boolean, similarity

def process_csv(file_path):
    """
    Reads a CSV file, processes the last two columns for similarity,
    and checks if the second column contains the prefix "Απαντηση: ".
    Adds two new columns: one for similarity (True/False) and another for similarity percentage.
    Returns the percentage of True results.
    """
    results = []
    true_count = 0
    total_count = 0

    output_rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header if it exists
        headers.extend(["Similarity", "Similarity Percentage"])

        for row in reader:
            # Assuming last two columns are to be compared
            col1 = row[-2] if len(row) > 1 else ""
            col2 = row[-1] if len(row) > 1 else ""

            # Remove prefix from the second column
            col2_cleaned = remove_prefix(col2)

            # Check similarity
            similar, similarity_percentage = is_similar(col1, col2_cleaned)

            # Track counts for percentage calculation
            total_count += 1
            if similar:
                true_count += 1

            # Append new columns to the row
            row.extend([similar, round(similarity_percentage, 2)])
            output_rows.append(row)

    new_filename = add_prefix_filename_from_path(file_path,"evaluated_")
    # Write updated rows to a new CSV
    with open(new_filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(output_rows)

    # Return percentage of True results
    true_percentage = (true_count / total_count) * 100 if total_count > 0 else 0
    return true_percentage


print("precentage: ",process_csv('results/simple_prompt_1/output_llama3_1.csv'))

# Example usage
# file_path = "path_to_your_file.csv"
# results = process_csv(file_path)
# for row in results:
#     print(f"Col1: {row[0]}, Col2: {row[1]}, Similar: {row[2]}")
