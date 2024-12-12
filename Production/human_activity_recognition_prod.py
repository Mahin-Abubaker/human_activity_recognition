import json
import random
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import argparse
import mlflow
class Experiment:

    def __init__(self):
        self.file_path = None
        self.data = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.model = LogisticRegression(max_iter=10000)  # Initialize logistic regression model

    def get_dataset_path_from_argument(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True, help='human activity dataset path')
        args = parser.parse_args()
        self.file_path=args.dataset
        mlflow.autolog()
        # mlflow.log_param("hello_param", "action_classifier")

    def load_data(self):
        try:
            selected_columns = ['sub', 'act','alx','aly','alz']
            self.data = pd.read_csv(self.file_path,usecols=selected_columns)
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
            required_columns = ["subject", "activity"]
            if all(column in self.data.columns for column in required_columns):
            # Remove rows where 'subject' or 'activity' has NaN
                self.data = self.data.dropna(subset=required_columns)
            print("Rows with NaN in 'subject' or 'activity' columns have been removed.")

        if self.data is not None:
            for column in self.data.columns:
                if self.data[column].isnull().sum() > 0:
                    if strategy == "mean":
                        self.data[column] = self.data[column].fillna(self.data[column].mean())                       
                    elif strategy == "median":
                        self.data[column] = self.data[column].fillna(self.data[column].median())
                    elif strategy == "mode":
                        self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
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
            categorical_columns = self.data.select_dtypes(include=['category','object']).columns
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
                # Drop the 'subject' column and use only relevant columns as features
                self.X = self.data.drop(columns=[target_column, 'subject'])  # Drop 'subject' as well
                self.y = self.data[target_column]
                print(f"Features and labels split successfully. Target column: {target_column}")
            else:
                print(f"Error: Target column '{target_column}' not found in dataset.")
        else:
            print("Dataset is not loaded yet.")

    def split_train_test(self, test_size=0.5, random_state=0):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)
            print(f"Dataset split into training and testing sets. Test size: {test_size}")
        else:
            print("Features and labels are not defined. Ensure to split features and labels first.")

    def scale_features(self):
        if self.x_train is not None and self.x_test is not None:
            numeric_columns = self.x_train.select_dtypes(include=['float64', 'int64']).columns
            # Initialize the scaler
            scaler = StandardScaler()
        
            # Apply scaling to the numeric data
            self.x_train[numeric_columns] = scaler.fit_transform(self.x_train[numeric_columns])
            self.x_test[numeric_columns] = scaler.transform(self.x_test[numeric_columns])           
            print("Features scaled successfully.")
        else:
            print("Training and testing sets are not defined. Ensure to split the dataset first.")

    def filter_activities(self, activities):
        print(self.data)
        if self.data is not None:
            if "activity" in self.data.columns:                            
                self.data = self.data[self.data["activity"].isin(activities)]
                self.data["activity"] = self.data["activity"].apply(
                    lambda x: x if x in activities else "0"
                )
                activity_counts = self.data["activity"].value_counts()  
                print(f"Counts of selected activities: \n{activity_counts}")             
            else:
                print("The 'activity' column does not exist in the dataset.")                
        else:
            print("Dataset is not loaded yet.")

    def train_logistic_regression(self):
        if self.x_train is not None and self.y_train is not None:
            # Fit the logistic regression model
            self.model.fit(self.x_train, self.y_train)
            print("Logistic Regression model trained successfully.")              
        else:
            print("Training data is not available. Please split the data first.")

    def evaluate_model(self):
        if self.x_test is not None and self.y_test is not None:
            # Predict the labels using the test set
            y_pred = self.model.predict(self.x_test)
            # Evaluate the model
            accuracy = accuracy_score(self.y_test, y_pred)
            accuracy_percentage = accuracy * 100
            print(f"Accuracy: {accuracy_percentage:.2f}%")
            mlflow.log_metric("Accuracy", accuracy_percentage) 
            # Calculate AUC
            Y_scores = self.model.predict_proba(self.x_test)
            #print(Y_scores)
            auc = roc_auc_score(self.y_test, Y_scores, multi_class='ovr')
            print(f'AUC: {auc}')
            mlflow.log_metric(f'Model_AUC:', auc)

            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
        else:
            print("Test data is not available. Please split the data first.")

    def random_prediction(self, location):
        # Predict the class label for the given location in the test set
        output_class = self.model.predict(self.X_test.iloc[[location]])  # Predict the class

        # Convert the output_class (array) to a string or log the actual label
        activity = output_class[0]  # Get the first (and only) prediction, assuming it's a single label

        # Print and log the predicted class
        print(f"Predicted class: {activity}")
        
        # Log the predicted class label as a string or an integer (if needed)
        mlflow.log_metric(f'Predicted class label:', activity)  # Can log as integer or string

        
    # def cross_validate_model(self):
        # if self.X is not None and self.y is not None:
        #     # Perform cross-validation
        #     scores = cross_val_score(self.model, self.X, self.y, cv=5)
        #     print(f"Cross-validation scores: {scores}")
        #     print(f"Mean accuracy: {np.mean(scores) * 100:.2f}%")
        # else:
        #     print("Data is not available for cross-validation. Please split the data first.")


if __name__ == "__main__":
    
    # Create an instance of Experiment
    experiment = Experiment()
    
    #Set dataset path
    experiment.get_dataset_path_from_argument()
    # Load the dataset
    experiment.load_data()

    #rename columns
    experiment.rename_columns({'act': 'activity', 'sub': 'subject'})    

    # # # Remove duplicate rows
    experiment.remove_duplicates()    

    # Check and handle missing values mean, median or mode
    experiment.handle_missing_values("mean")#Here I have used mean.
   
    #Fix the datatypes of the columns which are in correct
    column_type_mapping = {
        'subject': 'string'  
    }
    experiment.fix_incorrect_datatypes(column_type_mapping) 

    # Select 4 to 5 activities for the prediction
    #Here I am selecting walking, jogging, cycling,jumping,sitting
    #You can add more items here if you want more
    selected_activities = ["Jogging", "Walking", "Jumping", "Cycling", "Sitting"]
    experiment.filter_activities(selected_activities)

    #Encode categorical data
    experiment.encode_categorical_data()
    
    # Split the dataset into features and labels
    target_column = 'activity'  
    experiment.split_features_labels(target_column=target_column)
    
    # Split the dataset into training and testing sets
    experiment.split_train_test(test_size=0.2)
    
    # Scale the features
    experiment.scale_features()

    # Train logistic regression model
    experiment.train_logistic_regression()

    # Evaluate the model
    experiment.evaluate_model()

    # Make a random prediction
    experiment.random_prediction(random.randint(0, 3000))
# #kcrossfull validation
#52874332-1ce3-4393-a130-2534bbfd30f4 subscription id
