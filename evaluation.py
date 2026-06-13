import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(
            X_test
        )

        metrics = {

            "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

            "Precision":
            precision_score(
                y_test,
                y_pred,
                average="weighted"
            ),

            "Recall":
            recall_score(
                y_test,
                y_pred,
                average="weighted"
            ),

            "F1":
            f1_score(
                y_test,
                y_pred,
                average="weighted"
            )

        }

        return metrics

    def evaluate_all_models(
        self,
        models,
        X_test,
        y_test
    ):

        results = []

        for name, model in models.items():

            metric = self.evaluate_model(
                model,
                X_test,
                y_test
            )

            metric["Model"] = name

            results.append(
                metric
            )

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]
        ]

        return results_df

    def get_best_model(
        self,
        results_df
    ):

        best_model = (

            results_df

            .sort_values(
                by="F1",
                ascending=False
            )

            .iloc[0]

        )

        return best_model