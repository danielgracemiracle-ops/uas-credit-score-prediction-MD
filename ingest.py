import pandas as pd

class DataIngestion:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):

        df = pd.read_csv(
            self.filepath
        )

        print(
            f"Dataset loaded successfully"
        )

        print(
            f"Shape: {df.shape}"
        )

        return df

    def dataset_info(self, df):

        print("\nDataset Information")

        print(
            f"Rows : {df.shape[0]}"
        )

        print(
            f"Columns : {df.shape[1]}"
        )

    def preview_data(self, df):

        return df.head()