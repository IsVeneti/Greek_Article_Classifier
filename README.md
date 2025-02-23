# Greek Article Classifier

The Greek Article Classifier is a Python project that classifies Greek news articles using large language models (LLMs) with zero-shot prompts. In addition to its main classification functionality, the repository includes separate evaluation and baseline modules.

## Table of Contents
- [Greek Article Classifier](#greek-article-classifier)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Command Line Arguments](#command-line-arguments)
  - [Usage](#usage)
    - [Main Program](#main-program)
    - [Evaluation](#evaluation)
    - [BERT Baseline](#bert-baseline)
  - [License](#license)

## Overview

This project uses LLMs to perform zero-shot classification of Greek news articles. The main entry point (`main.py`) provides various command-line options to customize behavior, such as specifying a custom client, saving results, and adjusting logging or testing parameters.

## Installation

To set up and run the project, follow these steps:

1. **Download and Install Python**

   - Ensure you have **Python 3.8 or higher** installed. You can download it from [python.org](https://www.python.org/downloads/).
2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the LLM**

   ```bash
   python main.py --local --save results.csv --num_ctx 32768 --file_logs logs.log --random_testing_dataframe 20 --save_interval 10
   ```

## Command Line Arguments

When running `main.py`, the following arguments are available:

| Argument                       | Short    | Type     | Description                                                       |
| ------------------------------ | -------- | -------- | ----------------------------------------------------------------- |
| `--custom_client`            | `-cc`  | `str`  | Uses a custom client with the specified IP address.               |
| `--local`                    | `-l`   | `flag` | Runs using a local Ollama instance.                               |
| `--save`                     | `-s`   | `str`  | Specifies the file path where the results should be stored.       |
| `--num_ctx`                  | `-ctx` | `int`  | Sets a custom number of tokens for Ollama.                        |
| `--file_logs`                | `-fl`  | `str`  | Saves logs to the specified filename instead of stdout.           |
| `--random_testing_dataframe` | `-rtd` | `int`  | Runs on a randomized subset of the dataset of the specified size. |
| `--save_interval`            | `-si`  | `int`  | Saves results to the CSV file at the specified interval.          |

## Usage

### Main Program

Run the main script to classify articles using zero-shot prompts:

```bash
python main.py --local --save results.csv --num_ctx 32768 --file_logs logs.log --save_interval 50
```

### Evaluation

To run the full evaluation of the classifier, execute:

```bash
python src/evaluation/full_eval.py
```

### BERT Baseline

For comparison purposes, a BERT-based baseline classifier is provided. Run it with:

```bash
python src/bert/bert.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
