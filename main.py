
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.merge(pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/train_features.csv'),
                pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/train_labels.csv'))

test = pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/test_features.csv')

sample_submission = pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/SubmissionFormat.csv')

# Splitting the Data

train, val = train_test_split(train, train_size)