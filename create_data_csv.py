from article_data_processing import create_dataframe


DATA_FOLDER = "data2"

data_df = create_dataframe(DATA_FOLDER)

data_df.to_csv('data_csv_file.csv',sep='|', encoding='utf-8',index=False)

