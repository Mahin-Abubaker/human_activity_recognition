import unittest
import tempfile
import os
import mlflow
import argparse
from train import trainModel
from sklearn.linear_model import LogisticRegression


class TestTrain(unittest.TestCase):

    def setUp(self):
        # Define expected classes and feature dimensions
        self.classes = ["Jogging", "Walking", "Jumping", "Cycling", "Sitting"]
        self.numFeatures = 12

    def test_pipelineToModel(self):
        """
        Integration test for the ML pipeline. Verifies the end-to-end process
        from data loading to model training, saving, and reloading.
        """
        # Use a temporary directory to isolate the test and avoid filesystem pollution
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train the model using the `trainModel` function
            model, trainX, trainY = trainModel("train.csv", os.path.join(tmpdir, "model"))

            # Verify training data integrity
            self.assertEqual(len(trainX), len(trainY), "Mismatch in the number of features and labels.")
            for i in range(len(trainX)):
                self.assertEqual(len(trainX[i]), self.numFeatures, f"Feature length mismatch at row {i}.")
                self.assertIn(trainY[i], self.classes, f"Unexpected class label '{trainY[i]}' at row {i}.")
                self.assertIn(model.predict([trainX[i]])[0], self.classes, f"Model prediction not in allowed classes.")

            # Verify model properties
            self.assertIsInstance(model, LogisticRegression, "Model is not of type LogisticRegression.")
            self.assertEqual(model.coef_.shape[1], self.numFeatures, "Model coefficients do not match feature size.")

            # Verify model saving
            model_path = os.path.join(tmpdir, "model", "model.pkl")
            self.assertTrue(os.path.exists(model_path), "Model file not found at expected path.")

            # Verify model loading
            loadedModel = mlflow.sklearn.load_model(os.path.join(tmpdir, "model"))
            self.assertIsInstance(loadedModel, LogisticRegression, "Loaded model is not of type LogisticRegression.")
            self.assertEqual(model.coef_.shape, loadedModel.coef_.shape, "Loaded model coefficients do not match.")

if __name__ == '__main__':
    unittest.main()
