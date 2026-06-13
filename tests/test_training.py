from ingest import DataIngestion
from preprocessing import DataPreprocessor
from training import ModelTrainer

# ====================
# Load Data
# ====================

ingest = DataIngestion(
    "data_A.csv"
)

df = ingest.load_data()

# ====================
# Preprocess
# ====================

preprocessor = DataPreprocessor()

df = preprocessor.preprocess(df)

# ====================
# Train
# ====================

trainer = ModelTrainer()

X_train, X_test, y_train, y_test = (

    trainer.split_data(df)

)

models = trainer.train_models(
    X_train,
    y_train
)

print(
    "\nTraining Finished"
)

print(
    models.keys()
)