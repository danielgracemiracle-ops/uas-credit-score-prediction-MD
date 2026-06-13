from ingest import DataIngestion
from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator

# =====================
# Load
# =====================

ingest = DataIngestion(
    "data_A.csv"
)

df = ingest.load_data()

# =====================
# Preprocess
# =====================

preprocessor = DataPreprocessor()

df = preprocessor.preprocess(df)

# =====================
# Train
# =====================

trainer = ModelTrainer()

X_train, X_test, y_train, y_test = (

    trainer.split_data(df)

)

models = trainer.train_models(
    X_train,
    y_train
)

# =====================
# Evaluate
# =====================

evaluator = ModelEvaluator()

results = (

    evaluator.evaluate_all_models(
        models,
        X_test,
        y_test
    )

)

print("\nModel Comparison")

print(results)

print("\nBest Model")

print(
    evaluator.get_best_model(
        results
    )
)