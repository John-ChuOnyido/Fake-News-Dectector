import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading data from the system file 
# to get the shape and heads
necessity = pd.read_csv("Desktop/doc/story.txt", "r")

necessity.shape
necessity.head()

# Getting the labels
labels=necessity.label
labels.head()