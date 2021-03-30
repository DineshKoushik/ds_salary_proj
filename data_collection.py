# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:42:05 2021

@author: deshp
"""

import glassdoor_scraper as gs
import pandas as pd

path = "D:/chromedriver_win32/chromedriver"

df = gs.get_jobs("data_scientist", 100, False, path, 5)

df.to_csv('glassdoor_jobs.csv', index = False)
