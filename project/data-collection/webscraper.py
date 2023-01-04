# Program to scrape youtube music for
# songs with a corresponding insturmental track
# used to gather new traning data

from ytmusicapi import YTMusic
import billboard
from dataclasses import dataclass
import pandas as pd
import numpy as np
import datetime
import hopsworks
from utils import Song

### Global Variables ###
MIN_DIFF = 10


class Scraper:
    def __init__(self):
        self.chart = billboard.ChartData("hot-100")
        self.yt = YTMusic("secrets/headers_auth.json")
        # Use if for non-auth --->  self.yt = YTMusic()
        self.df = pd.DataFrame(columns=["title", "artist", "url", "inst_url", "diff"])

    def _get_songs(self):
        for song in self.chart:
            yield Song(song.title, song.artist)

    def scrape(self):
        for song in self._get_songs():
            found_songs = self._get_song(song)

            if len(found_songs) > 0:
                found_insts = self._get_instrumental(song, found_songs)

            if len(found_insts) > 0:
                best_match = self._find_closest_match(found_songs, found_insts)

                self.df = pd.concat(
                    [
                        self.df,
                        pd.DataFrame(
                            {
                                "title": [best_match.title],
                                "artist": [best_match.artist],
                                "url": [best_match.url],
                                "inst_url": [best_match.inst_url],
                                "diff": [best_match.diff],
                            }
                        ),
                    ]
                )

    def _get_song(self, song: Song) -> pd.DataFrame:
        df = pd.DataFrame(columns=["title", "artist", "url", "duration"])
        query = f"{song.title} {song.artist}"
        results = self.yt.search(query=query, filter="songs")
        print(f"SEARCH QUERY {query}")
        for result in results:
            title = result["title"]
            # TODO: Allow better artist matching
            artist = result["artists"][0]["name"]  # <--- Potential bug
            if title == song.title and artist == song.artist:
                print(f"FOUND SONG {song}")
                videoId = result["videoId"]
                url = f"https://www.youtube.com/watch?v={videoId}"
                try:
                    duration = datetime.datetime.strptime(result["duration"], "%M:%S")
                except:
                    duration = datetime.datetime.strptime(
                        result["duration"], "%H:%M:%S"
                    )
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "title": [song.title],
                                "artist": [song.artist],
                                "url": [url],
                                "duration": [duration],
                            }
                        ),
                    ]
                )

        return df

    def _get_instrumental(self, song: Song, found_songs: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(columns=["title", "artist", "url", "duration"])
        query = f"{song.title} {song.artist} instrumental"
        results = self.yt.search(query=query, filter="songs", limit=50)
        for result in results:
            if self._is_match(song, result, found_songs):
                print(f"FOUND INSTRUMENTAL {song}")
                videoId = result["videoId"]
                url = f"https://www.youtube.com/watch?v={videoId}"
                try:
                    duration = datetime.datetime.strptime(result["duration"], "%M:%S")
                except:
                    duration = datetime.datetime.strptime(
                        result["duration"], "%H:%M:%S"
                    )
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "title": [song.title],
                                "artist": [song.artist],
                                "url": [url],
                                "duration": [duration],
                            }
                        ),
                    ]
                )
        return df

    def _is_match(self, song, result, found_songs):
        title = result["title"]
        artist = result["artists"][0]["name"]
        videoId = result["videoId"]
        url = f"https://www.youtube.com/watch?v={videoId}"
        if (
            artist == song.artist
            and url not in found_songs["url"].values
            and "instrumental" in title.lower()
            and song.title.lower() in title.lower()
        ):
            return True

        return False

    def _find_closest_match(
        self, found_songs: pd.DataFrame, found_insts: pd.DataFrame
    ) -> Song:
        """
        Finds the closest match in duration between
        the found songs and found instrumentals
        """
        closest_match = Song(
            found_songs.iloc[0]["title"], found_songs.iloc[0]["artist"]
        )

        # Create a large time delta to compare against
        lowest_diff = datetime.timedelta(hours=1)

        for song in found_songs.itertuples():
            for inst in found_insts.itertuples():
                diff = abs(song.duration - inst.duration)
                if diff < lowest_diff:
                    lowest_diff = diff
                    closest_match.url = song.url
                    closest_match.inst_url = inst.url
                    closest_match.diff = lowest_diff

        return closest_match

    def upload_to_hopsworks(self):
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Add event_time column to data frame
        time_now = datetime.datetime.now()
        timestamps = np.full((len(self.df), 1), time_now, dtype=datetime.datetime)
        self.df["timestamp"] = timestamps

        # Drop index column
        self.df.reset_index(drop=True, inplace=True)

        # Upload to Hopsworks
        fg = fs.get_or_create_feature_group(
            name="youtube_music_data_from_billboard",
            version=1,
            primary_key=["title", "artist"],
            description="Links to songs and instrumentals from youtube music",
            event_time="timestamp",
        )

        try:
            fg.insert(self.df)
        except:
            print("Error uploading to Hopsworks")


if __name__ == "__main__":
    scraper = Scraper()
    scraper.scrape()
    scraper.upload_to_hopsworks()
