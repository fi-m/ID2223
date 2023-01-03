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


@dataclass
class Song:
    title: str
    artist: str
    # genre: str
    duration: str = None
    url: str = None
    inst_url: str = None

    def __str__(self):
        return f"{self.title} by {self.artist}"

    def __repr__(self):
        return f"{self.title} by {self.artist}"


class Scraper:
    def __init__(self):
        self.chart = billboard.ChartData("hot-100")
        self.yt = YTMusic("secrets/headers_auth.json")
        self.df = pd.DataFrame(columns=["title", "artist", "url", "inst_url"])

    def _get_songs(self):
        for song in self.chart:
            yield Song(song.title, song.artist)

    def scrape(self):
        for song in self._get_songs():
            found_song = self._get_song(song)

            if found_song:
                found_inst = self._get_instrumental(found_song)

            if found_inst:
                self.df = pd.concat(
                    [
                        self.df,
                        pd.DataFrame(
                            {
                                "title": [found_song.title],
                                "artist": [found_song.artist],
                                "url": [found_song.url],
                                "inst_url": [found_inst.inst_url],
                            }
                        ),
                    ]
                )
                print(self.df)

    def _get_song(self, song):
        query = f"{song.title} {song.artist}"
        results = self.yt.search(query=query, filter="songs")
        print(f"SEARCH QUERY {query}")
        for result in results:
            title = result["title"]
            # TODO: Allow better artist matching
            artist = result["artists"][0]["name"]  # <--- Potential bug
            if title == song.title and artist == song.artist:
                # print(f"FOUND SONG {song}")
                # __import__("pprint").pprint(result)
                videoId = result["videoId"]
                song.url = f"https://www.youtube.com/watch?v={videoId}"
                # print(f"URL: {song.url}")
                song.duration = result["duration"]
                return song

        return None

    def _get_instrumental(self, song):
        query = f"{song.title} {song.artist} instrumental"
        results = self.yt.search(query=query, filter="songs")
        for result in results:
            title = result["title"]
            artist = result["artists"][0]["name"]
            duration = result["duration"]
            videoId = result["videoId"]
            url = f"https://www.youtube.com/watch?v={videoId}"
            if (
                artist == song.artist
                # and duration == song.duration
                and url != song.url
                and "instrumental" in title.lower()
            ):
                print(f"FOUND INSTRUMENTAL {song}")
                # __import__("pprint").pprint(result)
                print(f"INSTRUMENTAL URL: {url}")
                song.inst_url = url
                return song

        return None

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
            name="youtube_music_data",
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
