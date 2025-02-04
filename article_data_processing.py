import os
import random
import pandas as pd

def create_dataframe(data_folder: str) -> pd.DataFrame:
    """
    Creates a dataframe from the text files in the given data folder.

    Args:
        data_folder (str): Path to the folder containing subfolders of text files.

    Returns:
        pd.DataFrame: A dataframe with columns 'title', 'article', and 'category'.
    """
    data_rows = []

    for category in os.listdir(data_folder):
        category_path = os.path.join(data_folder, category)
        if os.path.isdir(category_path):  # Ensure it is a subfolder
            for file_name in os.listdir(category_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(category_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    data_rows.append({"title": file_name, "article": content, "category": category})

    return pd.DataFrame(data_rows)

def create_dataframe_random(data_folder: str, num_rows: int) -> pd.DataFrame:
    """
    Creates a dataframe and selects a random subset of rows while ensuring a minimum row distance of 3.
    
    Args:
        data_folder (str): Path to the folder containing subfolders of text files.
        num_rows (int): The number of rows to select randomly.
    
    Returns:
        pd.DataFrame: A dataframe containing the selected rows.
    """
    df = create_dataframe(data_folder)
    if df.empty or num_rows <= 0:
        return pd.DataFrame(columns=df.columns)
    
    available_indices = list(range(len(df)))
    selected_indices = []
    
    while len(selected_indices) < num_rows and available_indices:
        index = random.choice(available_indices)
        selected_indices.append(index)
        # Remove nearby indices to enforce minimum distance of 3
        available_indices = [i for i in available_indices if abs(i - index) > 2]
    
    return df.iloc[selected_indices].reset_index(drop=True)

def count_rows_by_name(df, name_column_index=2):
    """
    Counts the number of rows for each unique name in the specified column.

    Args:
        df: The DataFrame to analyze.
        name_column_index: The index of the column containing the names (default is 2).

    Returns:
        A dictionary where the keys are the unique names and the values are the corresponding row counts.
    """

    name_counts = df.groupby(df.columns[name_column_index]).size()
    return name_counts.to_dict()


# # name_counts = count_rows_by_name(df)
# # print(name_counts)
# # Path to your data folder
# data_folder = "data2"
# output_file = "output.csv"
# # Create the dataframe
# dataframe = create_dataframe(data_folder)
# print(dataframe)