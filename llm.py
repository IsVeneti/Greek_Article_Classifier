from ollama import chat
from ollama import ChatResponse
import ollama

import os
import pandas as pd
import asyncio
from ollama import AsyncClient
from ollama import Client, ChatResponse
from article_data_processing import create_dataframe



async def query_ollama_and_update_async(df: pd.DataFrame, output_file: str, model: str = 'ymertzanis/meltemi7b', save_interval: int = 50):
    """
    Queries Ollama asynchronously for each article in the dataframe and updates it with the response.

    Args:
        df (pd.DataFrame): The dataframe with 'title', 'article', and 'category'.
        output_file (str): The file path to save the updated dataframe as a CSV.
        model (str): The Ollama model to use for querying.
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    client = AsyncClient()
    tasks = []
    
    # Define the response column
    if 'response' not in df.columns:
        df['response'] = None

    # Create async tasks for querying Ollama
    async def process_row(idx, article):
        print(article)
        prompt = f"Θελω να μου πείς αν αυτός ο αριθμός είναι ζυγός ή μονος. Απάντα μόνο με τη λέξη ζυγός/μονός. Αριθμος: {article}"
        message = {'role': 'user', 'content': prompt}
        response = await client.chat(model=model, messages=[message])
        print(response['message']['content'])
        df.at[idx, 'response'] = response['message']['content']  # Update the dataframe with the response

    for idx, row in df.iterrows():
        if pd.isna(row['response']):  # Skip rows already processed
            tasks.append(process_row(idx, row['article']))

        # Process tasks in chunks and save intermittently
        if len(tasks) >= save_interval:
            await asyncio.gather(*tasks)
            df.to_csv(output_file, index=True)  # Save dataframe
            tasks = []  # Reset tasks

    # Process any remaining tasks
    if tasks:
        await asyncio.gather(*tasks)
        df.to_csv(output_file, index=True)  # Final save


def query_ollama_and_update_with_custom_client(
    df: pd.DataFrame, output_file: str, host: str, headers: dict, model: str = 'llama3.2', save_interval: int = 50
):
    """
    Queries Ollama synchronously for each article in the dataframe using a custom client and updates it with the response.

    Args:
        df (pd.DataFrame): The dataframe with 'title', 'article', and 'category'.
        output_file (str): The file path to save the updated dataframe as a CSV.
        host (str): The host address for the custom Ollama client.
        headers (dict): Headers for the custom Ollama client.
        model (str): The Ollama model to use for querying.
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    client = Client(host=host, headers=headers)

    # Define the response column
    if 'response' not in df.columns:
        df['response'] = None

    # Iterate over rows in the dataframe
    for idx, row in df.iterrows():
        if pd.isna(row['response']):  # Skip rows already processed
            article = row['article']
            print(article)
            prompt = f"Θελω να μου πείς αν αυτός ο αριθμός είναι ζυγός ή μονος. Απάντα μόνο με τη λέξη ζυγός ή μονός. Αριθμος: {article}"
            message = {'role': 'user', 'content': prompt}

            try:
                # Query Ollama with the custom client
                response: ChatResponse = client.chat(model=model, messages=[message])
                df.at[idx, 'response'] = response.message.content  # Update the dataframe with the response
                print(response.message.content)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

            # Save the dataframe every `save_interval` calls
            if idx % save_interval == 0 and idx > 0:
                df.to_csv(output_file, index=True)
                print(f"Saved progress at row {idx}.")

    # Final save after completing all rows
    df.to_csv(output_file, index=False)
    print("Final dataframe saved.")


def query_ollama_and_update(df: pd.DataFrame, output_file: str, model: str = 'llama3.2', save_interval: int = 50):
    """
    Queries Ollama synchronously for each article in the dataframe and updates it with the response.

    Args:
        df (pd.DataFrame): The dataframe with 'title', 'article', and 'category'.
        output_file (str): The file path to save the updated dataframe as a CSV.
        model (str): The Ollama model to use for querying.
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    # Define the response column
    if 'response' not in df.columns:
        df['response'] = None

    # Iterate over rows in the dataframe
    for idx, row in df.iterrows():
        if pd.isna(row['response']):  # Skip rows already processed
            article = row['article']
            print(article)
            prompt = f"Θελω να μου πείς αν αυτός ο αριθμός είναι ζυγός ή μονος. Απάντα μόνο με τη λέξη ζυγός/μονός. Αριθμος: {article}"
            message = {'role': 'user', 'content': prompt}

            try:
                # Query Ollama
                response: ChatResponse = chat(model=model, messages=[message])
                df.at[idx, 'response'] = response.message.content  # Update the dataframe with the response
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

            # Save the dataframe every `save_interval` calls
            if idx % save_interval == 0 and idx > 0:
                df.to_csv(output_file, index=True)
                print(f"Saved progress at row {idx}.")

    # Final save after completing all rows
    df.to_csv(output_file, index=True)
    print("Final dataframe saved.")

# Path to your data folder
data_folder = "data2"
model = "llama3.1"
output_file = f"results/output_{model}1.csv"
# Create the dataframe
# dataframe = create_dataframe(data_folder)
print("okay")
example_dict= {'title': [1, 2,3], 'article': [1, 2, 3], 'category':[1,2,3]}
print
example_df = pd.DataFrame(data=example_dict)
print("Let's start!!")
custom_host = "http://localhost:11434"  
header = {"Content-Type": "application/json"}
# Run the async function
query_ollama_and_update(example_df, output_file,model= model,save_interval=2)
print("Finished.")
