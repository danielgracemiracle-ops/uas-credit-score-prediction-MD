from ingest import DataIngestion
from preprocessing import DataPreprocessor

# Load data
ingest = DataIngestion(
    "data_A.csv"
)

df = ingest.load_data()

# Preprocess data
preprocessor = DataPreprocessor()

df_processed = preprocessor.preprocess(df)

print("\nProcessed Shape:")
print(df_processed.shape)

print("\nDataset Info:")
print(df_processed.info())

print("\nPreview:")
print(df_processed.head())