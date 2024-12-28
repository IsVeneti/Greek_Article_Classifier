import os
import csv
import re
from newspaper import Article
from urllib.parse import quote_plus

def fetch_article_content(url, folder, parent_folder="data6", language='el'):
    """
    Fetches article content from the given URL using newspaper4k
    and writes it to a text file named after the article's title in a subfolder.

    Parameters:
        url (str): The URL of the article to fetch.
        folder (str): The subfolder where the article's text file will be saved.
        parent_folder (str): The base directory where all articles are saved. Default is "data".
        language (str): The language of the article. Default is 'el' for Greek.
    """
    try:
        # Properly encode the URL to handle any special characters (like Greek letters)
        encoded_url = quote_plus(url, safe="%/:=&?~#+!$,;'@()*[]-")

        # Initialize the article with the encoded URL and set language
        article = Article(encoded_url, language=language)

        # Download and parse the article
        article.download()
        article.parse()

        # Extract the title and content
        title = article.title or "Untitled"
        content = article.text or "No Content Available"

        # Clean the title to create a valid file name
        title_cleaned = re.sub(r'[\/:*?"<>|]', '', title)
        folder_path = os.path.join(parent_folder, folder)
        output_file = os.path.join(folder_path, f"{title_cleaned}.txt")

        # Ensure the folder structure exists
        os.makedirs(folder_path, exist_ok=True)

        # Save to a file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"{title}\n\n")
            file.write(content)
        
        print(f"Article content saved to {output_file}")

    except Exception as e:
        print(f"An error occurred while processing URL {url}: {e}")

def read_csv_to_object(csv_file):
    """
    Reads a CSV file and converts it into a list of dictionaries.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        list[dict]: List of dictionaries with 'url' and 'category' keys.
    """
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    data.append({'url': row[0], 'category': row[1]})
                else:
                    print(f"Skipping row with insufficient data: {row}")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
    return data

def process_articles(data, parent_folder="data6", language='el'):
    """
    Processes articles from a list of dictionaries and saves them to specified folders.

    Parameters:
        data (list[dict]): List of dictionaries containing 'url' and 'category' keys.
        parent_folder (str): The base directory where all articles will be saved. Default is "data".
        language (str): The language of the articles. Default is 'el' for Greek.
    """
    for item in data:
        url = item.get('url')
        folder = item.get('category')
        if url and folder:
            fetch_article_content(url, folder, parent_folder=parent_folder, language=language)

# if __name__ == "__main__":
#     # Path to the CSV file
#     csv_file_path = "text_categorization.csv"

#     # Step 1: Read CSV into an object
#     csv_data = read_csv_to_object(csv_file_path)

#     # Step 2: Process articles
#     process_articles(csv_data,parent_folder="data2")
