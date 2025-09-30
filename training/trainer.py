import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

class EEGTrainer:
    def __init__(self, model, threshold=0.5, weights=None):
        self.model = model
        self.threshold = threshold
        self.weights = weights

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(classification_report(y_test, y_pred, digits=3))
        return acc, f1
