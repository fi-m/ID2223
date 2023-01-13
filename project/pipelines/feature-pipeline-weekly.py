import sys
import pandas as pd
from tqdm import tqdm
import spec_utils
import soundfile as soundfile
import numpy as np
import datetime
import librosa
import hopsworks
import os

sys.path.append("../data-collection")
from youtube_downloader import YoutubeDownloader

class FeaturePipelineWeekly:
    def __init__(self, sr=22050, hop_length=512, n_fft=1024):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.inst_dir = "../data-collection/instruments/"
        self.mix_dir = "../data-collection/mixtures/"

    def download_from_youtube(self):
        yd = YoutubeDownloader()
        yd.download()
        yd.trim_files()

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
            ret.append([X_cache_path, y_cache_path, coef, artist, title])

        return ret

    def convert_to_df(self, training_set):
        X = []
        y = []
        coef = []
        artist = []
        title = []

        for (
            x,
            y_,
            coef_,
            artist_,
            title_,
        ) in training_set:
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
        inst_path = "TrainigData/Instrumentals/"
        mix_path = "TraningData/Mixtures/"

        project = hopsworks.login()
        dataset_api = project.get_dataset_api()

        # print column names
        x_hops = []
        y_hops = []
        print("Uploading to hopsworks...")
        success = False
        while not success:
            try:
                for row in tqdm(df.itertuples()):
                    # upload mixtures
                    mix_path = "TrainingData/mixtures"
                    x_hops.append(
                        mix_path + "/" + row.artist + " - " + row.title + ".npy"
                    )
                    dataset_api.upload(row.x, mix_path, overwrite=True)

                    # upload instrumentals
                    inst_path = "TrainingData/instrumentals"
                    y_hops.append(
                        inst_path + "/" + row.artist + " - " + row.title + ".npy"
                    )
                    dataset_api.upload(row.y, inst_path, overwrite=True)
                success = True
            except:
                print("Failed to upload, retrying...")

        fs = project.get_feature_store()
        df["x"] = x_hops
        df["y"] = y_hops

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
            primary_key=["artist", "title"],
            description="Processed data from Billboard",
            event_time="timestamp",
        )

        fg.insert(df)


if __name__ == "__main__":
    # Create a FeaturePipelineWeekly object
    pipeline = FeaturePipelineWeekly()
    # Run the pipeline
    pipeline.download_from_youtube()
    t = pipeline.run()
    # Convert to a pandas dataframe
    df = pipeline.convert_to_df(t)
    # Upload to Hopsworks
    pipeline.upload_to_hopsworks(df)
