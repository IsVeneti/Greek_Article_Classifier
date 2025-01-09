import os


def replace_slash_dots_with_underscore(string_with_slash):
    clean_string = string_with_slash.replace('/','_')
    clean_string = string_with_slash.replace('.','_')

    return clean_string


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

        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)

        print(f"File '{file_name}' successfully created in folder '{folder_name}'.")
    except OSError as e:
        print(f"Error creating folder or file: {e}")
        raise

def get_filename_from_path(filepath):
    return os.path.basename(filepath)


def add_prefix_filename_from_path(filepath,filename_prefix):
    path, filename = os.path.split(filepath)
    print(filename)
    # filename = os.path.splitext(filename)[0]
    new_filename = f'{filename_prefix}_{filename}'
    newpath = os.path.join(path, new_filename)
    return newpath

print(add_prefix_filename_from_path("C:/docojs/baby.csv","prefixbitch"))