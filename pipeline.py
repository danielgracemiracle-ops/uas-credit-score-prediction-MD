from ingest import DataIngestion
from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CreditScoreExperiment")

import joblib


class TrainingPipeline:

    def __init__(self):

        self.ingest = DataIngestion(
            "data_A.csv"
        )

        self.preprocessor = (
            DataPreprocessor()
        )

        self.trainer = (
            ModelTrainer()
        )

        self.evaluator = (
            ModelEvaluator()
        )

    def run(self):

        # ====================
        # Load
        # ====================

        df = self.ingest.load_data()

        # ====================
        # Preprocess
        # ====================

        df = self.preprocessor.preprocess(
            df
        )

        # ====================
        # Split
        # ====================

        X_train, X_test, y_train, y_test = (

            self.trainer.split_data(df)

        )

        # ====================
        # Train
        # ====================

        models = (

            self.trainer.train_models(
                X_train,
                y_train
            )

        )

        # ====================
        # Evaluate
        # ====================

        results = (

            self.evaluator
            .evaluate_all_models(
                models,
                X_test,
                y_test
            )

        )

        print("\nResults")

        print(results)

        # ====================
        # MLflow Logging
        # ====================

        for name, model in models.items():
    
            metrics = self.evaluator.evaluate_model(
                model,
                X_test,
                y_test
            )

            with mlflow.start_run(run_name=name):
            
                mlflow.log_param(
                    "model_name",
                    name
                )

                mlflow.log_metrics({
                    "accuracy": metrics["Accuracy"],
                    "precision": metrics["Precision"],
                    "recall": metrics["Recall"],
                    "f1_score": metrics["F1"]
                })

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model"
                )

        # ====================
        # Best Model
        # ====================

        best_row = (

            self.evaluator
            .get_best_model(
                results
            )

        )

        best_model_name = (
            best_row["Model"]
        )

        best_model = (
            models[best_model_name]
        )

        print(
            f"\nBest Model: {best_model_name}"
        )

        # ====================
        # Save Model
        # ====================
        
        joblib.dump(
            best_model,
            "best_model.pkl"
        )
        
        joblib.dump(
            self.preprocessor,
            "preprocessor.pkl"
        )
        
        print(
            "Model saved as best_model.pkl"
        )
        
        print(
            "Preprocessor saved as preprocessor.pkl"
        )
        
        return best_model