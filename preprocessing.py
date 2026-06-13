import pandas as pd
import numpy as np
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    def __init__(self):

        self.target_encoder = LabelEncoder()
        self.feature_encoders = {}

    # ==================================================
    # Feature Selection
    # ==================================================

    def feature_selection(self, df):

        drop_cols = [
            'Unnamed: 0',
            'ID',
            'Customer_ID',
            'Name',
            'SSN'
        ]

        df = df.drop(
            columns=drop_cols
        )

        return df

    # ==================================================
    # Data Cleaning
    # ==================================================

    def clean_numeric_columns(self, df):

        object_numeric_cols = [

            'Age',
            'Annual_Income',
            'Num_of_Loan',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Outstanding_Debt',
            'Amount_invested_monthly',
            'Monthly_Balance'

        ]

        for col in object_numeric_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(
                    "_",
                    "",
                    regex=False
                )
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    # ==================================================
    # Invalid Value Handling
    # ==================================================

    def handle_invalid_values(self, df):

        # Age

        df.loc[
            (df["Age"] < 18) |
            (df["Age"] > 100),
            "Age"
        ] = np.nan

        # Num Bank Accounts

        df.loc[
            df["Num_Bank_Accounts"] > 20,
            "Num_Bank_Accounts"
        ] = np.nan

        # Num Credit Card

        df.loc[
            df["Num_Credit_Card"] > 20,
            "Num_Credit_Card"
        ] = np.nan

        # Interest Rate

        df.loc[
            df["Interest_Rate"] > 100,
            "Interest_Rate"
        ] = np.nan

        # Num Loan

        df.loc[
            (df["Num_of_Loan"] < 0) |
            (df["Num_of_Loan"] > 20),
            "Num_of_Loan"
        ] = np.nan

        # Monthly Balance

        df.loc[
            df["Monthly_Balance"] < 0,
            "Monthly_Balance"
        ] = np.nan

        return df

    # ==================================================
    # Missing Value Handling
    # ==================================================

    def handle_missing_values(self, df):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        categorical_cols = df.select_dtypes(
            exclude=np.number
        ).columns

        num_imputer = SimpleImputer(
            strategy="median"
        )

        cat_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        df[numeric_cols] = (
            num_imputer.fit_transform(
                df[numeric_cols]
            )
        )

        df[categorical_cols] = (
            cat_imputer.fit_transform(
                df[categorical_cols]
            )
        )

        return df

    # ==================================================
    # Feature Engineering
    # ==================================================

    def convert_history_age(self, text):

        if pd.isna(text):
            return np.nan

        match = re.search(
            r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?',
            str(text)
        )

        if match:

            years = int(
                match.group(1)
            )

            months = int(
                match.group(2)
            )

            return years * 12 + months

        return np.nan

    def feature_engineering(self, df):

        df["Credit_History_Months"] = (

            df["Credit_History_Age"]

            .apply(
                self.convert_history_age
            )

        )

        df["Debt_to_Income_Ratio"] = (

            df["Outstanding_Debt"]

            /

            df["Annual_Income"]

        )

        df.drop(
            columns=["Credit_History_Age"],
            inplace=True
        )

        return df

    # ==================================================
    # Encoding
    # ==================================================

    def encode_features(self, df):

        # Target

        df["Credit_Score"] = (

            self.target_encoder

            .fit_transform(
                df["Credit_Score"]
            )

        )

        # Features

        categorical_cols = (

            df.select_dtypes(
                include="object"
            )

            .columns

        )

        for col in categorical_cols:

            le = LabelEncoder()

            df[col] = (

                le.fit_transform(
                    df[col].astype(str)
                )

            )

            self.feature_encoders[col] = le

        return df

    # ==================================================
    # Full Pipeline
    # ==================================================

    def preprocess(self, df):

        print("Feature Selection...")
        df = self.feature_selection(df)

        print("Cleaning Data...")
        df = self.clean_numeric_columns(df)

        print("Handling Invalid Values...")
        df = self.handle_invalid_values(df)

        print("Handling Missing Values...")
        df = self.handle_missing_values(df)

        print("Feature Engineering...")
        df = self.feature_engineering(df)

        print("Encoding Features...")
        df = self.encode_features(df)

        print("Preprocessing Finished")

        return df
    
    