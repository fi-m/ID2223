import subprocess
import os
import hopsworks
import pandas as pd
from utils import Song
import librosa
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import librosa.display


class YoutubeDownloader:
    def __init__(self):
        self.df = self._get_df_from_hopsworks()
        self.test_dir = "instruments/"
        self.train_dir = "mixtures/"

    def _get_df_from_hopsworks(self):
        # Get Feature Store
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Get feature view
        # Can filter out songs that are above certain diff
        fg = fs.get_feature_group("youtube_music_data_from_billboard", version=1)
        df = fg.read()
        return df

    def download(self):
        # Make test and train directories
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        for song in self._get_song():
            self._download(song, song.url, self.train_dir)
            self._download(song, song.inst_url, self.test_dir)

    def _get_song(self):
        # yield one row from the dataframe
        for _, row in self.df.iterrows():
            yield Song(
                artist=row["artist"],
                title=row["title"],
                url=row["url"],
                inst_url=row["inst_url"],
            )

    def _download(self, song, url=None, PATH=None):
        subprocess.call(
            [
                "yt-dlp",
                "-f",
                "bestaudio",
                "-x",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "--add-metadata",
                "-o",
                f"{PATH}{song.artist} - {song.title}.%(ext)s",
                url,
            ]
        )

    def trim_files(self):
        """
        Function to trim the audio files to the same length
        """
        for inst_wav, song_wav, song_name in self._get_files():
            # Before trim
            print("SONG NAME: ", song_name)
            print("BERFORE TRIM")
            print("LENGTH OF SONG: ", librosa.get_duration(y=song_wav))
            print("LENGTH OF INST: ", librosa.get_duration(y=inst_wav))
            self._wav_to_spectrogram(song_wav, song_name)
            self._wav_to_spectrogram(inst_wav, song_name + " inst")

            # Trim
            song_wav, inst_wav = self._trim(song_wav, inst_wav)

            # After trim
            print("AFTER TRIM")
            print("LENGTH OF SONG: ", librosa.get_duration(y=song_wav))
            print("LENGTH OF INST: ", librosa.get_duration(y=inst_wav))

    def _get_files(self):
        for file in os.listdir(self.test_dir):
            if file.endswith(".wav"):
                yield (
                    librosa.load(self.test_dir + file)[0],
                    librosa.load(self.train_dir + file)[0],
                    file,
                )

    def _wav_to_spectrogram(self, wav, song_name):
        try:
            plt.cla()
        except:
            pass

        # Convert to spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(D, y_axis="linear", x_axis="time")
        plt.colorbar(format="%+2.0f dB")
        plt.title(song_name)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()

    def _trim(self, song, inst):
        """
        Function to trim the audio files to the same length

        Parameters
        ----------
        song : np.array
        inst : np.array

        Returns
        -------
        song : np.array
        inst : np.array
        """

        # TODO: Make the function

        return song, inst


if __name__ == "__main__":
    yd = YoutubeDownloader()
    yd.download()
    yd.trim_files()
