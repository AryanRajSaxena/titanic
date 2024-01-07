import yaml
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import pathlib

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def split_data(data, split_size):
    train, test = train_test_split(data, test_size=split_size, random_state=42)
    return train, test

def save_data(train, test, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + "/train.csv", index = False)
    test.to_csv(output_path + "/test.csv", index = False)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dirr = curr_dir.parent.parent.parent
    # input_file = sys.argv[1] #raw

    # data_path = home_dirr.as_posix() +  input_file
    data_path = home_dirr.as_posix() +  "/data/raw/titanic.csv"
    output_path = home_dirr.as_posix() + "/data/processed"

    params_file = home_dirr.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))['make_dataset']

    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_size'])

    save_data(train_data, test_data, output_path)

if __name__ == '__main__':
    main()

