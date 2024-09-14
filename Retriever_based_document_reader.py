# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 08:44:54 2024

@author: Arun Joshi
"""

# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv 
# loading variables from .env file
load_dotenv() 
 
print(os.getenv("GEMINI_API_KEY"))

