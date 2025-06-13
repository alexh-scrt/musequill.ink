"""
env.py - load the project environment
"""
import sys
from dotenv import load_dotenv, find_dotenv

if not load_dotenv(find_dotenv()):
    print('❌  FAILED TO LOAD THE ENV')
    sys.exit(1)
