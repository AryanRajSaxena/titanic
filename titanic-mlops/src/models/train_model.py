import pathlib
import yaml
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(train_features,target, n_estimators, max_depth):
    model = RandomForestClassifier( n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(train_features, target)
    return model

def save_model(model, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,output_path + "/model.joblib")

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))['train_model']

    # input_file = sys.argv[1]
    data_path = home_dir.as_posix() + "/data/processed"
    output_path = home_dir.as_posix() + "/model"

    train_features = pd.read_csv(data_path + "/train.csv")
    x = train_features.iloc[1:,2:]
    y = train_features.iloc[1:,1]

    train_models = train_model(x, y, params['n_estimators'], params['max_depth'])

    save_model(train_models,output_path)

if __name__ == "__main__":
    main()
