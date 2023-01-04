import re
import os
from dataclasses import dataclass

@dataclass
class Song:
    title: str
    artist: str
    # genre: str
    duration: str = None
    url: str = None
    inst_url: str = None
    diff = None

    def __str__(self):
        return f"{self.title} by {self.artist}"

    def __repr__(self):
        return f"{self.title} by {self.artist}"


def save_files(insturment: str = None, full: str = None):
    pass


def clean_file_name(file_name: str = None):
    """
    Clean the filename of a given file.
    The the file name is cleaned by removing all non-alphanumeric characters

    params:
        file_name: str, the name of the file to be cleaned

    returns:
        str, the cleaned file name
    """
    cleaned_file_name = re.sub(
        "( ?\\((?!feat|ft|mix|remix|dj)(.*)\\) ?)*|( ?\\[(?!feat|FEAT|Feat|ft|mix|remix|dj)(.*)\\] ?)*|(\\.mp3)$",
        "",
        file_name.lower(),
    )

    return cleaned_file_name
