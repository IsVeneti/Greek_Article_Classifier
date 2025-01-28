import logging
import os
from pathlib import Path
from ollama import chat
from ollama import ChatResponse

import pandas as pd
from ollama import Client, ChatResponse
import yaml

from article_data_processing import create_dataframe
logger = logging.getLogger(__name__)

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
    logger.info(model)

    if response_column not in df.columns:
        df[response_column] = None

    # Iterate over rows in the dataframe
    for idx in range(len(df)):
        if pd.isna(df.loc[idx, response_column]):  # Skip rows already processed
            # Format the prompt using the provided template
            prompt = f"{prompt_template}\n{df.loc[idx, 'article']}"
            message = {'role': 'user', 'content': prompt}

            try:
                # Query Ollama
                response: ChatResponse = chat(model=model, messages=[message])
                logger.info(f"Response: {response}")
                df.loc[idx, response_column] = response.message.content  # Update the dataframe with the response
            except Exception as e:
                logger.info(f"Error processing row {idx}: {e}")
                logger.error("Error processing row {idx}: {e}")
                continue

            # Save the dataframe every `save_interval` calls
            if idx % save_interval == 0 and idx > 0:
                df.to_csv(output_file, index=False, sep='|')
                logger.info(f"Saved progress at row {idx}.")

    # Final save after completing all rows
    df.to_csv(output_file, index=False,sep='|')
    logger.info("Final dataframe saved.")

def query_custom_client_and_update(df: pd.DataFrame, output_file: str, model: str, host_ip: str, headers: dict = None, prompt_template: str = "Prompt", save_interval: int = 50):
    """
    Queries Ollama using a custom client for each article in the dataframe and updates it with the response.

    Args:
        df (pd.DataFrame): The dataframe with 'title', 'article', and 'category'.
        output_file (str): The file path to save the updated dataframe as a CSV.
        model (str): The Ollama model to use for querying.
        host_ip (str): The custom host IP address.
        headers (dict): Optional headers for the client.
        prompt_template (str): The template for the prompt, where {article} is replaced with the article text.
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    # Initialize the Ollama client
    client = Client(host=host_ip, headers=headers or {})

    # Define the model-specific response column
    response_column = model
    logger.info(f"Using model: {model} on host: {host_ip}")

    if response_column not in df.columns:
        df[response_column] = None

    # Iterate over rows in the dataframe
    for idx in range(len(df)):
        if pd.isna(df.loc[idx, response_column]):  # Skip rows already processed
            # Format the prompt using the provided template
            prompt = f"{prompt_template}\n{df.loc[idx, 'article']}"
            message = {'role': 'user', 'content': prompt}

            try:
                # Query Ollama using the client
                response = client.chat(model=model, messages=[message])
                logger.info(f"Response: {response}")
                df.loc[idx, response_column] = response["message"]["content"]  # Update the dataframe with the response
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

            # Save the dataframe every `save_interval` calls
            if idx % save_interval == 0 and idx > 0:
                df.to_csv(output_file, index=False, sep='|')
                logger.info(f"Saved progress at row {idx}.")

    # Final save after completing all rows
    df.to_csv(output_file, index=False, sep='|')
    logger.info("Final dataframe saved.")



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

            logger.info(f"File '{file_name}' successfully created in folder '{folder_name}'.")
    except OSError as e:
        logger.error(f"Error creating folder or file: {e}")
        raise


def replace_slash_dots_with_underscore(string_with_slash):
    clean_string = string_with_slash.replace('/','_')
    clean_string = string_with_slash.replace('.','_')

    return clean_string

def run_prompt_from_yaml(yaml_file: str, dataframe: pd.DataFrame, folder: str, save_interval: int = 50):
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
        logger.info(f"Processing model: {model_name} ({model})")

        # Prepare a DataFrame for prompts
        for prompt_name, prompt in prompts.items():
            # print(prompt_name)
            save_string_to_folder(f"{folder}/{prompt_name}", f"{prompt_name}.txt",prompt)
            output_file = f"{folder}/{prompt_name}/output_{model_name}.csv"
            # Call the query function
            query_ollama_and_update(
                df=dataframe.copy(),
                output_file=output_file,
                model=model,
                prompt_template=prompt,
                save_interval=save_interval
            )

def run_prompt_from_yaml_cc(yaml_file: str, dataframe: pd.DataFrame, folder: str, host_ip: str ,save_interval: int = 50):
    """
    Reads a YAML file containing prompts and models, and runs each prompt for every model in a custom client.

    Args:
        yaml_file (str): Path to the YAML file containing prompts and models.
        dataframe (pd.DataFrame): The dataframe with the data
        host_ip (str): The IP of the host where the model will run
        save_interval (int): The number of API calls after which to save the dataframe.
    """
    # Load the YAML file
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    prompts = config.get('prompts', {})
    models = config.get('models', {})

    # Iterate over each model
    for model_name, model in models.items():
        logger.info(f"Processing model: {model_name} ({model})")

        # Prepare a DataFrame for prompts
        for prompt_name, prompt in prompts.items():
            # logger.info(prompt_name)
            save_string_to_folder(f"{folder}/{prompt_name}", f"{prompt_name}.txt",prompt)
            output_file = f"{folder}/{prompt_name}/output_{model_name}.csv"
            # Call the query function
            query_custom_client_and_update(
                df=dataframe.copy(),
                output_file=output_file,
                model=model,
                prompt_template=prompt,
                host_ip=host_ip,
                save_interval=save_interval
            )

# number = 3

# # Path to your data folder
# data_folder = "data2"
# model = "gemma2"

# prompt = "Παρακαλώ πες μου τι είδους άρθρο είναι αυτο που γραφω κατω, από τις επιλογες Αρθρο, Επιστολή, Κριτική, Συνέντευξη, Συνταγή, Άλλο. Απάντησε μόνο με μία λεξη, από τα είδη που ανεφερα."
# prompt_name = "simple_prompt_2"
# save_string_to_folder("results/"+prompt_name, prompt_name + ".txt",prompt)
# output_file = f"results/{prompt_name}/output_{replace_slash_dots_with_underscore(model)}.csv"

# # Create the dataframe
# dataframe = create_dataframe(data_folder)
# print("okay")
# # example_dict= {'title': [1, 2,3], 'article': [1, 2, 3], 'category':[1,2,3]}
# # example_df = pd.DataFrame(data=example_dict)
# print("Let's start!!")
# # custom_host = "http://localhost:11434"  
# header = {"Content-Type": "application/json"}
# # Run the async function
# # print(dataframe.tail(20))
# run_prompt_from_yaml("prompt_settings.yaml",dataframe,"updated_results",save_interval=30)
# # query_ollama_and_update(dataframe, output_file,model= model,prompt_template = prompt,save_interval=10)
# print("Finished.") 
