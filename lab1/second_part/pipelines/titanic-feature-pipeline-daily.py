"""
Add synthetic passenger data to feature store.
"""

################     Initate modal    ################
import modal

stub = modal.Stub("lab-1-titanic-upload-new-passenger")
image = modal.Image.debian_slim().pip_install(
    [
        "pandas",
        "numpy",
        "hopsworks",
    ]
)
######################################################


def generate_passenger():
    import hopsworks
    import numpy as np
    import pandas as pd
    import datetime

    # Init hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get old data
    fg = fs.get_feature_group(name="titanic_modal_features", version=1)
    df = pd.DataFrame(fg.read())

    # Generate new passenger
    passengerid = df["passengerid"].max() + 1
    survived = np.random.choice(df["survived"])
    pclass = np.random.choice(df["pclass"])
    # Generate sex based on survied
    sex = np.random.choice(df["sex"][df["survived"] == survived])
    age = np.random.choice(df["age"])
    sibsp = np.random.choice(df["sibsp"])
    parch = np.random.choice(df["parch"])
    embarked = np.random.choice(df["embarked"])
    # Generate title based on sex
    title = np.random.choice(df["title"][df["sex"] == sex])

    new_passenger = pd.DataFrame(
        [
            [
                passengerid,
                survived,
                pclass,
                sex,
                age,
                sibsp,
                parch,
                embarked,
                title,
                datetime.datetime.now(),
            ]
        ],
        columns=[
            "passengerid",
            "survived",
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "embarked",
            "title",
            "timestamp",
        ],
    )

    print("New passenger: \n", new_passenger)
    return new_passenger


@stub.function(
    image=image,
    schedule=modal.Period(hours=20),
    secret=modal.Secret.from_name("titanic-secret"),
)
def main():
    import hopsworks

    # Init hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Create feature group
    fg = fs.get_or_create_feature_group(
        name="titanic_modal_features",
        version=1,
        primary_key=["passengerid"],
        description="Processed Titanic Dataset",
        event_time="timestamp",
    )

    new_passenger = generate_passenger()
    fg.insert(new_passenger)


if __name__ == "__main__":
    stub.deploy("lab-1-titanic-upload-new-passenger")
