
# Author: 'Ringo SW CHU"

import os
from icrawler.builtin import GoogleImageCrawler

CLASSES = ["OOCL_VESSEL_SHIPS", "MAERSK_VESSEL_SHIPS"]
NUM_SAMPLES = 50


for c in CLASSES:

    if not os.path.exists(os.path.join(os.getcwd(), c)):
        os.mkdir(c)

    google_crawler = GoogleImageCrawler(storage={'root_dir': c})
    google_crawler.crawl(keyword=c, max_num=NUM_SAMPLES)