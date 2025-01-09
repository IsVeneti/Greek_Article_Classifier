import os
from pathlib import Path
from ollama import chat
from ollama import ChatResponse

import pandas as pd
import asyncio
from ollama import AsyncClient
from ollama import Client, ChatResponse
import yaml

from article_data_processing import create_dataframe



async def query_ollama_and_update_async(df: pd.DataFrame, output_file: str, model: str, save_interval: int = 50):
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




def query_ollama_and_update(df: pd.DataFrame, output_file: str, model: str, prompt_template: str = "Prompt", save_interval: int = 50):
    """
    Queries Ollama synchronously for each article in the dataframe and updates it with the response.

    Args:
        df (pd.DataFrame): The dataframe with 'title', 'article', and 'category'.
        output_file (str): The file path to save the updated dataframe as a CSV.
        model (str): The Ollama model to use for querying.
        prompt_template (str): The template for the prompt, where {article} is replaced with the article text.
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    # Define the model-specific response column
    response_column = model
    print(model)

    if response_column not in df.columns:
        df[response_column] = None

    # Iterate over rows in the dataframe
    for idx in range(len(df)):
        if pd.isna(df.loc[idx, response_column]):  # Skip rows already processed
            # Format the prompt using the provided template
            prompt = prompt_template.format(article=df.loc[idx, 'article'])
            message = {'role': 'user', 'content': prompt}

            try:
                # Query Ollama
                response: ChatResponse = chat(model=model, messages=[message])
                print(f"Response: {response}")
                df.loc[idx, response_column] = response.message.content  # Update the dataframe with the response
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

            # Save the dataframe every `save_interval` calls
            if idx % save_interval == 0 and idx > 0:
                df.to_csv(output_file, index=False)
                print(f"Saved progress at row {idx}.")

    # Final save after completing all rows
    df.to_csv(output_file, index=False)
    print("Final dataframe saved.")

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

def save_string_to_folder(folder_name: str, file_name: str, file_content: str):
    """
    Creates a folder and saves a string in a text file inside the folder.

    Args:
        folder_name (str): The name of the folder to be created.
        file_name (str): The name of the text file to be created inside the folder.
        file_content (str): The content to be written into the text file.

    Raises:
        OSError: If the folder or file cannot be created due to an operating system error.
    """
    try:
        # Create the folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)

        # Define the full path for the file
        file_path = os.path.join(folder_name, file_name)
        file_path = Path(file_path)
        if not file_path.exists():
            # Write the content to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(file_content)

            print(f"File '{file_name}' successfully created in folder '{folder_name}'.")
    except OSError as e:
        print(f"Error creating folder or file: {e}")
        raise


def replace_slash_dots_with_underscore(string_with_slash):
    clean_string = string_with_slash.replace('/','_')
    clean_string = string_with_slash.replace('.','_')

    return clean_string

def run_prompt_from_yaml(yaml_file: str, dataframe: pd.DataFrame, save_interval: int = 50):
    """
    Reads a YAML file containing prompts and models, and runs each prompt for every model.

    Args:
        yaml_file (str): Path to the YAML file containing prompts and models.
        dataframe (pd.DataFrame): The dataframe with the data
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    # Load the YAML file
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    prompts = config.get('prompts', {})
    models = config.get('models', {})

    # Iterate over each model
    for model_name, model in models.items():
        print(f"Processing model: {model_name} ({model})")

        # Prepare a DataFrame for prompts
        for prompt_name, prompt in prompts.items():
            print(prompt_name)
            save_string_to_folder("results/"+prompt_name, prompt_name + ".txt",prompt)
            output_file = f"results/{prompt_name}/output_{replace_slash_dots_with_underscore(model)}.csv"
            # Call the query function
            query_ollama_and_update(
                df=dataframe,
                output_file=output_file,
                model=model,
                prompt_template=prompt,
                save_interval=save_interval
            )


def run_prompt_from_yaml1(yaml_file):


    save_string_to_folder("results/"+prompt_name, prompt_name + ".txt",prompt)
    output_file = f"results/{prompt_name}/output_{replace_slash_dots_with_underscore(model)}.csv"
    

number = 3

# Path to your data folder
data_folder = "data2"
model = "gemma2"

prompt = "Παρακαλώ πες μου τι είδους άρθρο είναι αυτο που γραφω κατω, από τις επιλογες Αρθρο, Επιστολή, Κριτική, Συνέντευξη, Συνταγή, Άλλο. Απάντησε μόνο με μία λεξη, από τα είδη που ανεφερα."
prompt_name = "simple_prompt_2"
save_string_to_folder("results/"+prompt_name, prompt_name + ".txt",prompt)
output_file = f"results/{prompt_name}/output_{replace_slash_dots_with_underscore(model)}.csv"

# Create the dataframe
dataframe = create_dataframe(data_folder)
print("okay")
# example_dict= {'title': [1, 2,3], 'article': [1, 2, 3], 'category':[1,2,3]}
# example_df = pd.DataFrame(data=example_dict)
print("Let's start!!")
# custom_host = "http://localhost:11434"  
header = {"Content-Type": "application/json"}
# Run the async function
# print(dataframe.tail(20))
run_prompt_from_yaml("prompt_settings.yaml",dataframe)
# query_ollama_and_update(dataframe, output_file,model= model,prompt_template = prompt,save_interval=10)
print("Finished.")
