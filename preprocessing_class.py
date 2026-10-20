import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SpaceshipPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_family_size=20):
        self.max_family_size = max_family_size
        self.family_sizes = None

    def fit(self, X, y=None):
        # Compute family sizes from the training data for safe lookup during transform
        df = X.copy()
        df['Name'] = df['Name'].fillna('Unknown')
        df['LastName'] = df['Name'].str.split().str[-1]
        self.family_sizes = df['LastName'].value_counts().to_dict()
        return self

    def transform(self, X):
        df = X.copy()

        # 1. Group ID & Cabin Forward/Backward fill
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        df['Cabin'] = df.groupby('Group')['Cabin'].bfill().ffill()

        # 2. Impute features based on Cabin uniformity
        cabin_categories = ['CryoSleep', 'VIP', 'HomePlanet', 'Destination']
        valid_cabin = df['Cabin'].notna()

        for col in cabin_categories:
            if col in df.columns:
                cabin_nunique = df[valid_cabin].groupby('Cabin')[col].transform(lambda x: x.nunique(dropna=True))
                uniform_cabin = valid_cabin & (cabin_nunique == 1) & df[col].isna()
                shared_cabin_values = df[valid_cabin].groupby('Cabin')[col].transform(lambda x: x.iloc[0] if x.notna().any() else None)
                df.loc[uniform_cabin, col] = shared_cabin_values[uniform_cabin]

        # 3. CryoSleep amenity zero-filling
        awake_amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in awake_amenity_cols:
            if col in df.columns and 'CryoSleep' in df.columns:
                df.loc[(df['CryoSleep'] == True) & df[col].isna(), col] = 0

        # 4. Family Name Parsing & Imputation
        df['HadName'] = df['Name'].notna().astype(int)
        df['Name'] = df['Name'].fillna('Unknown')
        df['LastName'] = df['Name'].str.split().str[-1]

        def get_cabin_lastname(group):
            valid_names = group[group != 'Unknown'].dropna()
            return valid_names.iloc[0] if not valid_names.empty else 'Unknown'

        if valid_cabin.any():
            cabin_family_name = df[valid_cabin].groupby('Cabin')['LastName'].transform(get_cabin_lastname)
            name_impute = valid_cabin & (df['LastName'] == 'Unknown') & (cabin_family_name != 'Unknown')
            df.loc[name_impute, 'LastName'] = cabin_family_name[name_impute]

        # Family-based feature imputation
        features_to_check = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
        for col in features_to_check:
            if col in df.columns and self.family_sizes:
                valid_family_mask = (
                    df['LastName'].notna() &
                    (df['LastName'] != 'Unknown') &
                    df['LastName'].map(lambda x: self.family_sizes.get(x, 0) <= self.max_family_size)
                )
                family_nunique = df[valid_family_mask].groupby('LastName')[col].transform(lambda x: x.nunique(dropna=True))
                uniform_families_mask = valid_family_mask & (family_nunique == 1) & df[col].isna()
                shared_family_values = df[valid_family_mask].groupby('LastName')[col].transform(
                    lambda x: x.iloc[0] if x.notna().any() else None
                )
                df.loc[uniform_families_mask, col] = shared_family_values[uniform_families_mask]

        # 5. Child and general spending zero-filling
        child_age_threshold = 18
        if 'Age' in df.columns:
            for col in awake_amenity_cols:
                if col in df.columns:
                    child_mask = (df['Age'] <= child_age_threshold) & df[col].isna()
                    df.loc[child_mask, col] = 0

        for col in awake_amenity_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 6. Cabin distribution parsing (Deck, Num, Side)
        if 'Cabin' in df.columns:
            cabin_split = df['Cabin'].str.split('/', expand=True)
            if cabin_split.shape[1] == 3:
                df['Deck'] = cabin_split[0]
                df['Num'] = pd.to_numeric(cabin_split[1], errors='coerce')
                df['Side'] = cabin_split[2]
            else:
                df['Deck'] = pd.Series(dtype='object')
                df['Num'] = pd.Series(dtype='float')
                df['Side'] = pd.Series(dtype='object')

        # 7. Drop unnecessary columns
        drop_cols = ['Cabin', 'PassengerId', 'Name']
        X_out = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # 8. Typecasting dtypes for XGBoost
        categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'LastName']
        for col in categorical_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype('category')

        bool_cols = ['CryoSleep', 'VIP']
        for col in bool_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype('boolean')

        if 'Group' in X_out.columns:
            X_out['Group'] = X_out['Group'].astype(int)

        return X_out
