import argparse
import logging

from article_data_processing import create_dataframe
from llm import run_prompt_from_yaml, run_prompt_from_yaml_cc

DATA_FOLDER = "data2"


parser = argparse.ArgumentParser(
                    prog='Greek_Article_Classifier',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument("-cc","--custom_client",type=str, help="Uses a custom client. Needs IP of the host.")
parser.add_argument("-l","--local",action="store_true", help="Uses a local ollama")
parser.add_argument("-s","--save",type=str, help="Where to store the results")

args = parser.parse_args()

dataframe = create_dataframe(DATA_FOLDER)
save_var = 'new_results'
logname = "logs.log"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

if args.save:
    save_var = args.save

if args.custom_client:
    print(args.custom_client)
    run_prompt_from_yaml_cc("prompt_settings.yaml",dataframe,save_var,args.custom_client,save_interval=30)


if args.local:
    run_prompt_from_yaml("prompt_settings.yaml",dataframe,save_var,save_interval=10)



