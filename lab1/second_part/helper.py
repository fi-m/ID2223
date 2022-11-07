import numpy as np
import pandas as pd
np.random.seed(seed=1337)

class Helper: 

    def get_age_from_title(df, title: str):
        """
        Helper function for filling missing ages in the data frame
        """
        mask = (df["Title"] == title) & df["Age"].isna()

        # Get sutible candidates for age sampling
        candidates = df.loc[(df["Title"] == title) & df["Age"].notna()]

        g = candidates.groupby("Age", dropna=True)["Age"].count()
        g = g.apply(lambda x: x/g.sum())

        weights = g.to_numpy()
        ages = g.index

        df.update(df["Age"][mask].apply(lambda x: np.random.choice(ages, p=weights)))
        
        print(df.head())

        return df
