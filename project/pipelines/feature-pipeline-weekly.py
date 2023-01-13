import os
import hopsworks
import librosa
import datetime
import numpy as np
import soundfile as soundfile
import spec_utils
from tqdm import tqdm
import pandas as pd


class FeaturePipelineWeekly:
    def __init__(self, sr=22050, hop_length=512, n_fft=1024):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.inst_dir = "../data-collection/instruments/"
        self.mix_dir = "../data-collection/mixtures/"

    def get_filelist(self):
        self.filelist = self._make_pairs()

    def _make_pairs(self):
        input_exts = [".wav", ".m4a", ".mp3", ".mp4", ".flac"]

        X_list = sorted(
            [
                os.path.join(self.mix_dir, fname)
                for fname in os.listdir(self.mix_dir)
                if os.path.splitext(fname)[1] in input_exts
            ]
        )
        y_list = sorted(
            [
                os.path.join(self.inst_dir, fname)
                for fname in os.listdir(self.inst_dir)
                if os.path.splitext(fname)[1] in input_exts
            ]
        )

        filelist = list(zip(X_list, y_list))

        return filelist

    def run(self):
        self.get_filelist()

        training_set = self.make_training_set(self.filelist)
        return training_set

    def make_training_set(self, filelist):
        ret = []
        for X_path, y_path in tqdm(filelist):
            X, y, X_cache_path, y_cache_path = spec_utils.cache_or_load(
                X_path, y_path, self.sr, self.hop_length, self.n_fft
            )
            coef = np.max([np.abs(X).max(), np.abs(y).max()])
            song_name = os.path.splitext(os.path.basename(X_path))[0]
            artist, title = song_name.split(" - ")
            ret.append([X, y, coef, artist, title])

        return ret

    def convert_to_df(self, training_set):
        X = []
        y = []
        coef = []
        artist = []
        title = []

        for x, y_, coef_, artist_, title_ in training_set:
            X.append(x)
            y.append(y_)
            coef.append(coef_)
            artist.append(artist_)
            title.append(title_)

        df = pd.DataFrame(
            {
                "x": X,
                "y": y,
                "coef": coef,
                "artist": artist,
                "title": title,
            }
        )

        return df

    def upload_to_hopsworks(self, df):
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Add event_time column to data frame
        time_now = datetime.datetime.now()
        timestamps = np.full((len(df), 1), time_now, dtype=datetime.datetime)
        df["timestamp"] = timestamps

        # Drop index column
        df.reset_index(drop=True, inplace=True)

        # Upload to Hopsworks
        fg = fs.get_or_create_feature_group(
            name="processed_youtube_music_data_from_billboard",
            version=1,
            primary_key=["title", "artist"],
            description="Processed data from Billboard",
            event_time="timestamp",
        )

        fg.insert(df)


if __name__ == "__main__":
    # Create a FeaturePipelineWeekly object
    pipeline = FeaturePipelineWeekly()
    # Run the pipeline
    t = pipeline.run()
    # Convert to a pandas dataframe
    df = pipeline.convert_to_df(t)
    # Upload to Hopsworks
    pipeline.upload_to_hopsworks(df)
