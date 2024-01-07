import pandas as pd
import joblib
import pathlib
import yaml
import sys

from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import tree
from dvclive import Live


def evaluate(model, x, y, split, live, save_path):
    prediction_by_class = model.predict_proba(x)
    predictions = prediction_by_class[:,1]

    avg_prec = metrics.average_precision_score(y, predictions)
    roc_score = metrics.roc_auc_score(y,predictions)

    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_score": {}}
    live.summary["avg_prec"][split] = avg_prec
    live.summary["roc_score"][split] = roc_score
    # live.log_metric("train/accuracy", avg_prec)

    live.log_sklearn_plot("roc", y, predictions, name = f"roc/{split}")

    live.log_sklearn_plot(
        "precision_recall",
        y,
        predictions,
        name = f"prc/{split}"
    )
    live.log_sklearn_plot(
        "confusion_matrix",
        y,
        prediction_by_class.argmax(-1),
        name=f"cm/{split}",
    )
    # live.next_step()

def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    model_file = home_dir.as_posix() + "/model/model.joblib"

    model = joblib.load(model_file)

    #load the data

    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    train_features = pd.read_csv(data_path + '/train.csv')
    x_train = train_features.iloc[1:,2:]
    y_train = train_features.iloc[1:,1]
    feature_names = x_train.columns.to_list()

    test_features = pd.read_csv(data_path + '/test.csv')
    x_test = test_features.iloc[1:,2:]
    y_test = test_features.iloc[1:,1]
    

    with Live(output_path, dvcyaml=False) as live:

        evaluate(model, x_train, y_train,"train",live, output_path )
        evaluate(model, x_test, y_test,"test",live, output_path )

        save_importance_plot(live, model, feature_names)


if __name__ == "__main__":
    main()