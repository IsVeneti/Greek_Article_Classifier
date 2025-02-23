import os
import pandas as pd
from pathlib import Path

def merge_evaluations(folder_path, output_file="../merged_evaluations.csv"):
    """
    Merges all CSV evaluation files in a folder into a single CSV.
    Adds 'prompt' and ensures multiple models per prompt are included.
    
    Parameters:
        folder_path (str): Path to the folder containing evaluation CSVs.
        output_file (str): Name of the output CSV file (default: "merged_evaluations.csv").
    
    Returns:
        pd.DataFrame: Merged DataFrame of all evaluations.
    """
    all_dfs = []
    
    # Iterate through all CSV files in the folder
    folder_path = Path(folder_path)
    for folder in os.listdir(folder_path):
        full_folder_path = os.path.join(folder_path, Path(folder))
        print(full_folder_path)
        for file in os.listdir(full_folder_path):
            print("FILE: ",file)
            if "evaluation" in file:
                file_path = os.path.join(full_folder_path, file)
                df = pd.read_csv(file_path,delimiter=',')
                
                # Extract prompt name from file name (without extension)

                file_path = Path(file_path)
                split_path = file_path.parts
                print(split_path)
                # eval_t = os.path.splitext(split_path[len(split_path)-1])[0]
                # df.insert(0, "evaluation_type", eval_t)  # Add prompt column
                prompt_name = split_path[len(split_path)-2]
                df.insert(0, "prompt_name", prompt_name)  # Add prompt column
                all_dfs.append(df)
    print(all_dfs)
    
    # Concatenate all dataframes into one
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_path, index=False)
    
    return merged_df

# Example usage:
merged_df = merge_evaluations("./results_13_02/results_32k","../../merged_evaluations.csv")
print(merged_df.head())
