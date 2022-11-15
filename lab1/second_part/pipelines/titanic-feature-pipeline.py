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
# cleanup_catergories = {"sex": {"female": 1, "male": 0}, "embarked": {"S": 0, "C": 1, "Q": 2}, "Cabin": {"N": 0, "C": 1, "E": 2, "G": 3, "D":4, "A": 5, "B": 6, "F": 7, "T": 8}}

sex_dict = {"female": 1, "male": 0}
embarked_dict = {"S": 0, "C": 1, "Q": 2}
cabin_dict = {"N": 0, "C": 1, "E": 2, "G": 3, "D":4, "A": 5, "B": 6, "F": 7, "T": 8}
cleanup_catergories = {"sex": sex_dict, "embarked": embarked_dict, "cabin": cabin_dict}

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
    import datetime
    # Load the data_frame
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
    )

    # Drop features and NaNs
    df.drop(["Ticket", "Fare"], axis=1, inplace=True)
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
    # Bin ages
    df['Age'] = pd.cut(df['Age'],[0,8,15,30,65,150])

    # Bin fare
    df['Fare'] = pd.cut(df['Fare'],[0,200,400,600,1000])
    
    
    # Bin SibSp
    pd.cut(df['SibSp'], [0,1,2,7], right=False)

    # Cabin into categories based on first letter(deck of boat)
    df["Cabin"] = df["Cabin"].str.slice(0,1)

    # Make a separate category of all te NANs
    df["Cabin"] = df["Cabin"].fillna("N")

    # Fixes for hopsworks...
    df.columns = df.columns.str.lower()

    # Final encoding
    df = df.replace(cleanup_catergories)

    # Add event_time column to data frame
    time_now = datetime.datetime.now()
    timestamps = np.full((len(df), 1), time_now, dtype=datetime.datetime)
    df["timestamp"] = timestamps

    # Load hopsworks project
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Create feature group
    fg = fs.create_feature_group(
        name="titanic_modal_features",
        version=1,
        primary_key=["passengerid"],
        description="Processed Titanic Dataset",
        event_time="timestamp",
    )

    print("LENGTH OF DATA FRAME", len(df))

    # Over write old data
    fg.insert(df, overwrite=True)


if __name__ == "__main__":
    with stub.run():
        main()
