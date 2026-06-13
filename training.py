from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier


class ModelTrainer:

    def __init__(self):

        self.models = {

            "RandomForest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

            "GradientBoosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            ),

            "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss"
            )

        }

    def split_data(self, df):

        X = df.drop(
            "Credit_Score",
            axis=1
        )

        y = df["Credit_Score"]

        X_train, X_test, y_train, y_test = (

            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    def train_models(
        self,
        X_train,
        y_train
    ):

        trained_models = {}

        for name, model in self.models.items():

            print(
                f"Training {name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            trained_models[name] = model

        return trained_models