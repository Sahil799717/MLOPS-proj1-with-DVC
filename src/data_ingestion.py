import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

log_dir = "logs"
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)-> pd.DataFrame:
    """Loading data"""
    try:
        df= pd.read_csv(data_url)
        logger.debug("Data loaded successfully from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Parsing error",e)
        raise
    except Exception as e:
        logger.error('Failed to load the data',e)

def preprocess_data(df:pd.DataFrame):
    """Preprocessing tha data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error during processing %s',e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path:str):
    """Saving train and test data to log directory"""
    try:
        raw_data_dir = os.path.join(data_path,'raw')
        os.makedirs(raw_data_dir, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_dir,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_dir,'test.csv'),index=False)
        logger.debug('data saved Successfully at %s',raw_data_dir)
    except Exception as e:
        logger.error('Error saving data %s',e)

def main():
    try:
        test_size = 0.2
        data_path = "https://raw.githubusercontent.com/Sahil799717/Datasets-all/refs/heads/main/spam.csv"
        df = load_data(data_url = data_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df, test_size = test_size, random_state = 42)
        save_data(train_data,test_data, data_path = './data')
    except Exception as e:
        logger.error('Error in main execution: %s', e)
        print('Error in main execution:', e)

if __name__ =='__main__':
    main()




