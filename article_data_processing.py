import os
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