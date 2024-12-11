import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Experiment:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
    
    def rename_columns(self, column_mapping):
        if self.data is not None:
            self.data.rename(columns=column_mapping, inplace=True)
            print(f"Columns renamed successfully: {column_mapping}")
        else:
            print("Dataset is not loaded yet. Cannot rename columns.")

    def remove_duplicates(self):
        if self.data is not None:
            before_count = len(self.data)
            self.data = self.data.drop_duplicates()
            after_count = len(self.data)
            print(f"Removed {before_count - after_count} duplicate rows. Remaining rows: {after_count}.")
        else:
            print("Dataset is not loaded yet. Cannot remove duplicates.")

    def handle_missing_values(self, strategy="mean"):
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print("Missing values in each column:")
            print(missing_values)
        if self.data is not None:
            for column in self.data.columns:
                if self.data[column].isnull().sum() > 0:
                    if strategy == "mean":
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    elif strategy == "median":
                        self.data[column].fillna(self.data[column].median(), inplace=True)
                    elif strategy == "mode":
                        self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            print(f"Missing values handled using {strategy} strategy.")
        else:
            print("Dataset is not loaded yet.")

    def show_dataset(self):
        if self.data is not None:           
            print(self.data)
        else:
            print("Dataset is not loaded yet.")

    def fix_incorrect_datatypes(self,column_type_mapping):
        if self.data is not None: 
            print("---------------------------------------Before fixing the datatypes------------------------")
            print(self.data.dtypes)
            for column, dtype in column_type_mapping.items():
                if column in self.data.columns:
                    self.data[column] = self.data[column].astype(dtype)
                    print(f"Column '{column}' converted to {dtype}")
                else:
                    print(f"Column '{column}' not found in the dataset.")

            print("---------------------------------------After fixing the datatypes------------------------")
            print(self.data.dtypes)
        else:
            print("Dataset is not loaded yet.") 
    
    def encode_categorical_data(self):
        if self.data is not None:
            categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
            if not categorical_columns.empty:
                print(f"Encoding categorical columns: {categorical_columns}")
                le = LabelEncoder()
                for col in categorical_columns:
                    self.data[col] = le.fit_transform(self.data[col])
            else:
                print("No categorical columns found to encode.")
        else:
            print("Dataset is not loaded yet.")

    def split_features_labels(self, target_column):
        if self.data is not None:
            if target_column in self.data.columns:
                self.X = self.data.drop(columns=[target_column])
                self.y = self.data[target_column]
                print(f"Features and labels split successfully. Target column: {target_column}")
            else:
                print(f"Error: Target column '{target_column}' not found in dataset.")
        else:
            print("Dataset is not loaded yet.")

    def split_train_test(self, test_size=0.2, random_state=42):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)
            print(f"Dataset split into training and testing sets. Test size: {test_size}")
        else:
            print("Features and labels are not defined. Ensure to split features and labels first.")

    def scale_features(self):
        if self.X_train is not None and self.X_test is not None:
            numeric_columns = self.X_train.select_dtypes(include=['float64', 'int64']).columns
            # Initialize the scaler
            scaler = StandardScaler()
        
            # Apply scaling to the numeric data
            self.X_train[numeric_columns] = scaler.fit_transform(self.X_train[numeric_columns])
            self.X_test[numeric_columns] = scaler.transform(self.X_test[numeric_columns])
            print(self.X_train)
            print("Features scaled successfully.")
        else:
            print("Training and testing sets are not defined. Ensure to split the dataset first.")

if __name__ == "__main__":
    # Provide the path to your dataset
    file_path = 'dataset.csv'
    
    # Create an instance of Experiment
    experiment = Experiment(file_path)
    
    # Load the dataset
    experiment.load_data()
    
    #rename columns
    experiment.rename_columns({'act': 'activity', 'sub': 'subject'})

    # # Remove duplicate rows
    experiment.remove_duplicates()

    print(experiment.data)
    
    # # Check and handle missing values mean, median or mode
    # experiment.handle_missing_values("mean")#Here I have used mean.

    # #Fix the datatypes of the columns which are in correct
    # column_type_mapping = {
    #     'time': 'datetime64[ns]',  # Convert 'time' to datetime type
    #     'gesture': 'category'          # Convert 'gesture' to categorical type
    # }
    # experiment.fix_incorrect_datatypes(column_type_mapping)

    # # Encode categorical data
    # experiment.encode_categorical_data()
    
    # # Split the dataset into features and labels
    # target_column = 'gesture'  # Updated target column name
    # experiment.split_features_labels(target_column=target_column)
    
    # # Split the dataset into training and testing sets
    # experiment.split_train_test(test_size=0.2)
    
    # # Scale the features
    # experiment.scale_features()

    # # Display the dataset after cleaning
    # experiment.show_dataset()
    # plt.figure(figsize=(10,8))
    # plt.title('Barplot of Activity')
    # sns.countplot(experiment.X_train.gesture)
    # plt.xticks(rotation=90)

# t['gesture'] = t['gesture'].replace({0: 'Move', 1: 'Calibrate', 2: 'Scan', 3: 'Return',4:'Idle'})
# print(t.head())
# plt.figure(figsize=(10,8))
# plt.title('Barplot of Activity')
# sns.countplot(t.gesture)
# plt.xticks(rotation=90)
# plt.show()
# #kcrossfull validation