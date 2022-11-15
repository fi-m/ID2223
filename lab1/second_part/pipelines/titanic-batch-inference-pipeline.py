###########     Initate modal    #####################
import modal

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

# @stub.function(image=image, secret=modal.Secret.from_name("titanic-secret"))


def main():
    import hopsworks
    import joblib
    import datetime
    import requests
    import os
    import pandas as pd
    import dataframe_image as dfi

    # Upload a batch of random data to hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get model from hopsworks
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal_simple_classifier", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    # Get batch data
    fv = fs.get_feature_view(name="titanic_modal_view", version=1)
    fv.init_batch_scoring(training_dataset_version=1)
    start_time = datetime.datetime.now() - datetime.timedelta(days=1)
    df = fv.get_batch_data()
    df = df[(df["timestamp"] >= start_time)]
    df.drop("timestamp", axis=1, inplace=True)
    print("BATCH LENGTH: ", len(df))

    # Generate predictions on new batch data
    y_pred = model.predict(df)
    print(y_pred[-1])

    # Last prediction
    latest_prediction = y_pred[-1]

    # Download funny images to upload
    os.makedirs("../resources/images/", exist_ok=True)
    if latest_prediction == 0:
        with open("../resources/images/latest_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://media.tenor.com/FghTtX3ZgbAAAAAC/drowning-leo.gif"
                ).content
            )

    else:
        with open("../resources/images/latest_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://media4.giphy.com/media/6A5zBPtbknIGY/giphy.gif?cid=ecf05e477syp5zeoheii45de76uicvgu0nuegojslz3zgodt&rid=giphy.gif&ct=g"
                ).content
            )

    print("PREDICTION: ", y_pred[-1])

    # Upload prediction
    dataset_api = project.get_dataset_api()
    dataset_api.upload(
        "../resources/images/latest_prediction.gif", "Resources/images", overwrite=True
    )

    # Get correct label and passengerid
    fg = fs.get_feature_group("titanic_modal_features", version=1)
    fg_df = fg.read()
    correct_label = fg_df["survived"].iloc[-1]
    passengerid = fg_df["passengerid"].iloc[-1]
    # Download correct gif
    if correct_label == 0:
        with open("../resources/images/correct_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://media.tenor.com/FghTtX3ZgbAAAAAC/drowning-leo.gif"
                ).content
            )

    else:
        with open("../resources/images/correct_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://media4.giphy.com/media/6A5zBPtbknIGY/giphy.gif?cid=ecf05e477syp5zeoheii45de76uicvgu0nuegojslz3zgodt&rid=giphy.gif&ct=g"
                ).content
            )

    # Uload correct label
    dataset_api.upload(
        "../resources/images/correct_prediction.gif", "Resources/images", overwrite=True
    )

    print("CORRECT: ", correct_label)

    # Get overview of batch data
    monitor_fg = fs.get_or_create_feature_group(
        name="titanic_predictions",
        version=1,
        primary_key=["passengerid"],
        description="Titanic Survival Prediction/Outcome Monitoring",
    )

    now = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        "prediction": [latest_prediction],
        "label": [correct_label],
        "datetime": [now],
        "passengerid": [passengerid],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)
    dfi.export(
        df_recent, "../resources/images/df_recent.png", table_conversion="matplotlib"
    )
    dataset_api.upload(
        "../resources/images/df_recent.png", "Resources/images", overwrite=True
    )

    predictions = history_df[["prediction"]]
    labels = history_df[["label"]]

    # TODO: CREATE HISTORTY AND UPLOAD IT


if __name__ == "__main__":
    main()
    # TODO
