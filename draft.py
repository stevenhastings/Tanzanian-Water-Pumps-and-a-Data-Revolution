
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.merge(pd.read_csv('C:\\Users\\steve\\WATERPUMPS\\Tanzanian-Water-Pumps-and-a-Data-Revolution\\train_features.csv'),
                pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/train_labels.csv'))

test = pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/test_features.csv')

sample_submission = pd.read_csv('Tanzanian-Water-Pumps-and-a-Data-Revolution/SubmissionFormat.csv')

# Splitting the Data

train, val = train_test_split(train, train_size=0.80, test_size=0.20,
                                stratify=train['status_group'], random_state=42)

print(f"train shape: {train.shape}, \n val shape: {val.shape}, \n test shape: {test.shape}")

