# import required libraries
import os
import csv
import pandas as pd

def create_meta_csv(dataset_path, destination_path=None):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """
    
    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)

    if not os.path.exists(os.path.join(destination_path, "dataset_attr.csv")):
        # Make a csv with full file path and labels
        for root, dirnames, files in os.walk(DATASET_PATH):
            with open(os.path.join(destination_path,'dataset_attr.csv'),'a') as new_file:
                csv_writer = csv.writer(new_file)
                for file in files:
                    csv_writer.writerow([os.path.join(root,file),os.path.basename(root)])
        

    # if no error
    return True

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    train_data = dframe.sample(frac=split_ratio)#,random_state=200)
    test_data  = dframe.drop(train_data.index)
### In return statement we are reseting the index number to match with row number for reducing confusion ###
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

def create_and_load_meta_csv_df(dataset_path, destination_path=None, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    # write out as dataset_attr.csv in destination_path directory
    # change destination_path to DATASET_PATH if destination_path is None
    if destination_path == None:
        destination_path = dataset_path 
    elif not os.path.exists(destination_path):
        destination_path = dataset_path
        print("meta file was stored in provided dataset_path")

    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'),names=["Path","Label"])

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe = dframe.sample(frac=1).reset_index(drop=True)

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe



if __name__ == "__main__":
    # test config
    dataset_path = '../Data/fruits'
    dest = '../Data/fruits'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())
