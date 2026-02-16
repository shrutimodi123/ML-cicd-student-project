import joblib 
import numpy as np

def predict_pass(study_hours,attendance,assignment_completed):
    model = joblib.load("models/model.pkl")

    features = np.array([[study_hours,attendance,assignment_completed]])
    prediction = model.predict(features)[0]

    return "Pass" if prediction ==1 else "Fail"

if __name__ == "__main__":
    result = predict_pass(6,80,7)
    print("Prediction:",result)