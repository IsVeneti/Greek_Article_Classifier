import argparse
import logging

from configs.config import DATA_DIR, PROMPT_SETTINGS_PATH
from src.data_processing.article_data_processing import create_dataframe, create_dataframe_random
from src.llms.ollama_connector import run_prompt_from_yaml, run_prompt_from_yaml_cc




parser = argparse.ArgumentParser(
                    prog='Greek_Article_Classifier',
                    description='This program runs llms in specific zero-shot prompts to classify news articles',
                    epilog='Have fun!')

parser.add_argument("-cc","--custom_client",type=str, help="Uses a custom client. Needs IP of the host.")
parser.add_argument("-l","--local",action="store_true", help="Uses a local ollama")
parser.add_argument("-s","--save",type=str, help="Where to store the results")
parser.add_argument("-ctx", "--num_ctx", type = int, help = "Custom number of tokens for ollama")
parser.add_argument("-fl","--file_logs",type=str, help="Returns the logs to the filename specified instead of stdout")
parser.add_argument("-rtd","--random_testing_dataframe",type=int, help="Test on a specific number of the dataset set by this parameter, randomized")
parser.add_argument("-si","--save_interval",type=int, help="Saves into the csv on intervals set by this paramets")


args = parser.parse_args()

dataframe = create_dataframe(DATA_DIR)
save_var = 'new_results'
logname = "logs.log"
num_ctx = 2048
save_interval = 50

if(args.random_testing_dataframe):
    dataframe = create_dataframe_random(DATA_DIR, num_rows=args.random_testing_dataframe)

logger = logging.getLogger(__name__)
if args.file_logs:
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

if args.save:
    save_var = args.save

if args.num_ctx:
    num_ctx = args.num_ctx

if args.save_interval:
    save_interval = args.save_interval

if args.custom_client:
    print(args.custom_client)
    run_prompt_from_yaml_cc(PROMPT_SETTINGS_PATH,dataframe,folder=save_var,num_ctx=num_ctx,host_ip=args.custom_client,save_interval=save_interval)


if args.local:
    run_prompt_from_yaml(PROMPT_SETTINGS_PATH,dataframe,folder=save_var,num_ctx=num_ctx,save_interval=save_interval)



