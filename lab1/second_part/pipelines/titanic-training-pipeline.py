"""
Slim-downed verion of the training part of the notebook
"""

################     Initate modal    ################
import modal
import xgboost as xgb

stub = modal.Stub("lab-1-titanic")
image = (
    modal.Image.debian_slim()
    .apt_install(["libgomp1"])
    .pip_install(
        [
            "pandas",
            "numpy",
            "hopsworks",
            "scikit-learn",
            "seaborn",
        ]
    )
)

######################################################


@stub.function(image=image, secret=modal.Secret.from_name("titanic-secret"))
def main():
    import os
    import hopsworks
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import plot_confusion_matrix
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    # Initiate hopworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    try:
        fv = fs.get_feature_view(name="titanic_modal_view", version=1)
    except:
        fg = fs.get_feature_group(name="titanic_modal_features", version=1)
        query = fg.select_except(["title", "passengerid"])
        fv = fs.create_feature_view(
            "titanic_modal_view",
            version=1,
            description="Read from Titanic Dataset",
            labels=["survived"],
            query=query,
        )
    # Create train_test_split
    X_train, X_test, y_train, y_test = fv.train_test_split(0.2)

    # Drop timestamp
    datasets = [X_train, X_test]
    for dataset in datasets:
        dataset.drop("timestamp", axis=1, inplace=True)

    # Init model
    # XGB
    # model = xgb.XGBClassifier(max_depth=3)
    model = DecisionTreeClassifier(random_state=1337)

    # Train
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    plot_confusion_matrix(model, X_test, y_test)

    # Upload to hopsworks
    mr = project.get_model_registry()

    # The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
    model_dir = "titanic_model"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/titanic_model.pkl")
    plt.savefig(model_dir + "/confusion_matrix.png")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    titanic_model = mr.python.create_model(
        name="titanic_modal_simple_classifier",
        version=1,
        metrics={"accuracy": metrics["accuracy"]},
        model_schema=model_schema,
        description="Titanic Survial-rate Predictions",
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    titanic_model.save(model_dir)


if __name__ == "__main__":
    with stub.run():
        main()
