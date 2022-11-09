"""
Slim-downed verion of the feature part of the notebook
"""

################     Initate modal    ################
import modal

stub = modal.Stub("lab-1-titanic")
image = modal.Image.debian_slim().pip_install(
    [
        "pandas",
        "numpy",
        "hopsworks",
    ]
)

################ Dicts with encodings ################
cleanup_catergories = {"sex": {"female": 1, "male": 0}, "embarked": {"S": 0, "C": 1, "Q": 2}}

sex_dict = {"female": 1, "male": 0}
embarked_dict = {"S": 0, "C": 1, "Q": 2}
# Reversed
"""
title_dict = {
    0: ["Mr"],
    1: ["Miss"],
    2: ["Mrs"],
    3: ["Master"],
    # Rare titles, not worth individual categorys
    4: [
        "Dr",
        "Rev",
        "Mlle",
        "Major",
        "Col",
        "Countess",
        "Capt",
        "Ms",
        "Sir",
        "Lady",
        "Nme",
        "Don",
        "Jonkheer",
    ],
}
"""
#####################################################


@stub.function(image=image, secret=modal.Secret.from_name("titanic-secret"))
def main():
    """
    Main function containg the feature engineering part
    of the pipeline.
    """
    import pandas as pd
    import numpy as np
    import hopsworks

    # Load the data_frame
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
    )

    # Drop features and NaNs
    df.drop(["Ticket", "Fare", "Cabin"], axis=1, inplace=True)
    df = df[df["Embarked"].notna()]

    # Feature engineering
    # Creat a title feature
    if "Name" in df.columns:
        df["Title"] = df.Name.str.extract("([A-Za-z]+)\\.")
        df.drop("Name", axis=1, inplace=True)

    # Interpolate missing ages
    for title in df["Title"].unique():
        # This sould be optimized
        mask = (df["Title"] == title) & df["Age"].isna()

        # Get sutible candidates for age sampling
        candidates = df.loc[(df["Title"] == title) & df["Age"].notna()]

        g = candidates.groupby("Age", dropna=True)["Age"].count()
        g = g.apply(lambda x: x / g.sum())

        weights = g.to_numpy()
        ages = g.index

        df.update(df["Age"][mask].apply(lambda x: np.random.choice(ages, p=weights)))

    # Cast age to int
    df["Age"] = df["Age"].astype("int")

    # Fixes for hopsworks...
    df.columns = df.columns.str.lower()

    # Final encoding
    df = df.replace(cleanup_catergories)

    # Load hopsworks project
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Create feature group
    fg = fs.get_or_create_feature_group(
        name="titanic_modal_features",
        # online_enabled=True,
        version=1,
        primary_key=["passengerid"],
        description="Processed Titanic Dataset",
    )

    # Have to overwrite since hopsworks validation blocks connection
    fg.insert(df, overwrite=True)


if __name__ == "__main__":
    with stub.run():
        main()
