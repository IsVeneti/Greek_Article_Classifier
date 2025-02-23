from configs.config import DATA_DIR
from src.data_processing.article_data_processing import create_dataframe



data_df = create_dataframe(DATA_DIR)

data_df.to_csv('data_csv_file.csv',sep='|', encoding='utf-8',index=False)

