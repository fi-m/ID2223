import unittest
import sys

sys.path.append("../data-collection")
from webscraper import Scraper

class TestScraper(unittest.TestCase):
    def test_scrape(self):
        scraper = Scraper()
        scraper.scrape()
        self.assertGreaterEqual(len(scraper.df), 1)


if __name__ == "__main__":
    unittest.main()
