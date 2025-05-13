import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df = df.dropna()

    X = df.drop(columns=['Class', 'Time'])  
    y = df['Class']

    # StandartScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # data split as train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE 
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler
