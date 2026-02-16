from src.train import train_model

def test_training():
    accuracy = train_model()
    assert accuracy >= 0.80