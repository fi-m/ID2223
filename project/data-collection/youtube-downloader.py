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
    def __init__(self, df = None):
        self.df = self._get_df_from_hopsworks()        
        self.inst_dir = "./dataset/instruments/"
        self.mix_dir = "./dataset/mixtures/"

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
        os.makedirs(self.inst_dir, exist_ok=True)
        os.makedirs(self.mix_dir, exist_ok=True)

        for song in self._get_song():
            self._download(song, song.url, self.mix_dir)
            self._download(song, song.inst_url, self.inst_dir)

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
        for Y_inst, Y_song, sr, song_name in self._get_files():
            # Before trim
            print("SONG NAME: ", song_name)
            print("BERFORE TRIM")
            print("LENGTH OF SONG: ", librosa.get_duration(y=Y_song, sr=sr))
            print("LENGTH OF INST: ", librosa.get_duration(y=Y_inst, sr=sr))
            # self._wav_to_spectrogram(Y_song, song_name)
            # self._wav_to_spectrogram(Y_inst, song_name + " inst")

            # Trim
            Y_song, Y_inst = self._trim(Y_song, Y_inst)

            # After trim
            print("AFTER TRIM")
            print("LENGTH OF SONG: ", librosa.get_duration(y=Y_song, sr=sr))
            print("LENGTH OF INST: ", librosa.get_duration(y=Y_inst, sr=sr))

            # Save the files
            self._save_files(Y_song, Y_inst, song_name, sr)

    def _get_files(self):
        for file in os.listdir(self.inst_dir):
            if file.endswith(".wav"):
                Y, sr = librosa.load(self.inst_dir + file, mono=False)
                X, _ = librosa.load(self.mix_dir + file, mono=False)
                yield (
                    Y,
                    X,
                    sr,
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

    def _trim(self, Y_song, Y_inst):
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
        # Get the length of the shortest audio file

        min_len_c1 = min(len(Y_song[0]), len(Y_inst[0]))
        min_len_c2 = min(len(Y_song[1]), len(Y_inst[1]))

        Y_trim_song_c1 = Y_song[0][:min_len_c1]
        Y_trim_song_c2 = Y_song[1][:min_len_c2]

        Y_trim_inst_c1 = Y_inst[0][:min_len_c1]
        Y_trim_inst_c2 = Y_inst[1][:min_len_c2]

        Y_song = np.array([Y_trim_song_c1, Y_trim_song_c2], dtype=float)
        Y_inst = np.array([Y_trim_inst_c1, Y_trim_inst_c2], dtype=float)

        # Trim the audio files
        return Y_song, Y_inst

    def _save_files(self, Y_song, Y_inst, song_name, sr):
        # Save the files
        sf.write(
            self.inst_dir + song_name,
            Y_inst.T,
            sr,
            subtype="PCM_24",
        )

        sf.write(
            self.mix_dir + song_name,
            Y_song.T,
            sr,
            subtype="PCM_24",
        )


if __name__ == "__main__":
    yd = YoutubeDownloader()
    yd.download()
    yd.trim_files()
