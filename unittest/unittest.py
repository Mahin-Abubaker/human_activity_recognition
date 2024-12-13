import unittest
import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment import Experiment  # Assuming your Experiment class is in the 'experiment.py' file


class TestExperiment(unittest.TestCase):

    def setUp(self):
        # Initialize the Experiment class
        self.experiment = Experiment()

        # Set the file path to a test dataset (replace with a valid test CSV file)
        self.experiment.file_path = 'test_data.csv'  # Make sure you have a test dataset in this path
        self.experiment.load_data()

        # Assume the 'activity' column is the target and the rest are features
        self.experiment.split_features_labels(target_column='activity')
        self.experiment.split_train_test(test_size=0.2)

    def test_load_data(self):
        # Test the dataset is loaded correctly
        self.assertIsNotNone(self.experiment.data, "Data should be loaded successfully.")
        self.assertGreaterEqual(len(self.experiment.data), 1, "Dataset should have at least one row.")
        self.assertIn('activity', self.experiment.data.columns, "'activity' column should be in the dataset.")
        self.assertIn('subject', self.experiment.data.columns, "'subject' column should be in the dataset.")

    def test_rename_columns(self):
        # Test renaming columns
        self.experiment.rename_columns({'act': 'activity', 'sub': 'subject'})
        self.assertIn('activity', self.experiment.data.columns, "'activity' column should exist after renaming.")
        self.assertIn('subject', self.experiment.data.columns, "'subject' column should exist after renaming.")

    def test_remove_duplicates(self):
        # Test removing duplicates
        before_count = len(self.experiment.data)
        self.experiment.remove_duplicates()
        after_count = len(self.experiment.data)
        self.assertLess(after_count, before_count, "Duplicates should have been removed.")
        self.assertEqual(len(self.experiment.data), len(self.experiment.data.drop_duplicates()), "Duplicates are not removed correctly.")

    def test_handle_missing_values(self):
        # Test handling missing values by filling with mean (you can modify strategy)
        self.experiment.handle_missing_values(strategy='mean')
        self.assertEqual(self.experiment.data.isnull().sum().sum(), 0, "There should be no missing values after handling.")

    def test_split_train_test(self):
        # Test splitting the data into train and test sets
        self.experiment.split_train_test(test_size=0.2)
        self.assertEqual(len(self.experiment.x_train) + len(self.experiment.x_test), len(self.experiment.X))
        self.assertEqual(len(self.experiment.y_train) + len(self.experiment.y_test), len(self.experiment.y))

    def test_scale_features(self):
        # Test scaling of features
        self.experiment.scale_features()
        # Check that features are scaled (mean ~0, std ~1)
        mean = self.experiment.x_train.mean().mean()
        std = self.experiment.x_train.std().mean()
        self.assertAlmostEqual(mean, 0, delta=1, msg="Features are not centered (mean ~= 0).")
        self.assertAlmostEqual(std, 1, delta=1, msg="Features are not scaled (std ~= 1).")

    def test_train_logistic_regression(self):
        # Test training the logistic regression model
        self.experiment.train_logistic_regression()
        self.assertIsInstance(self.experiment.model, LogisticRegression, "Model should be of type LogisticRegression.")

    def test_evaluate_model(self):
        # Test model evaluation (accuracy, confusion matrix, etc.)
        self.experiment.train_logistic_regression()
        accuracy = self.experiment.evaluate_model()
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be between 0 and 1.")
        self.assertLessEqual(accuracy, 1, "Accuracy should be between 0 and 1.")

    def test_random_prediction(self):
        # Test making a random prediction
        location = random.randint(0, len(self.experiment.x_test)-1)
        self.experiment.random_prediction(location)
        print("Random prediction test passed!")


if __name__ == '__main__':
    unittest.main()
