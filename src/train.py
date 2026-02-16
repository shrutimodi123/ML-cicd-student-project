import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    # Load dataset
    df = pd.read_csv("data/student_data.csv")

    X = df[["study_hours","attendance","assignments_completed"]]
    y = df["pass"]

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    print(f"Model Accuracy : {accuracy:.2f}")

    if accuracy< 0.80:
        raise ValueError("Accuracy below threshold(0.80)")
    
    os.makedirs("models",exist_ok=True)

    joblib.dump(model,"models/model.pkl")
    return accuracy

if __name__ == "__main__":
    train_model()

