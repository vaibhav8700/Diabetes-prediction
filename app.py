from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QGridLayout
from PyQt5.QtCore import Qt
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

df = pd.read_csv("..Documents/python/final_ui/app.py")

# Evaluating algorithms



def get_models():
    models = dict()

    # define the pipeline
    scaler = MinMaxScaler()
    power = PowerTransformer(method='yeo-johnson')

    clf1 = RandomForestClassifier()
    clf2 = CatBoostClassifier(verbose=False)
    clf3 = XGBClassifier()
    clf4 = LGBMClassifier()
    clf5 = LogisticRegression()

    models['Random Forest'] = Pipeline(
        steps=[('s', scaler), ('p', power), ('m', clf1)])
    models['Cat Boost'] = Pipeline(
        steps=[('s', scaler), ('p', power), ('m', clf2)])
    models['XGBoost'] = Pipeline(
        steps=[('s', scaler), ('p', power), ('m', clf3)])
    models['LightGBM'] = Pipeline(
        steps=[('s', scaler), ('p', power), ('m', clf4)])
    models['Logistic Regression'] = Pipeline(
        steps=[('s', scaler), ('p', power), ('m', clf5)])

    return models

# evaluate a given model using cross-validation


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
    scores = cross_val_score(
        model, X, y, scoring='f1_weighted', cv=cv, n_jobs=-1)
    return scores


# define dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
y = y.values.ravel()

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    scores = scores
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


# define the pipeline
model = RandomForestClassifier()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)

# Defining your search space
hyperparameter_space = {
    "n_estimators": [25, 50, 75],
    "criterion": ["gini"],
    "max_depth": [3, 5, 10, None],
    "class_weight": ["balanced"],
    "min_samples_split": [0.001, 0.01, 0.05, 0.1],
}


clf = GridSearchCV(model, hyperparameter_space,
                   scoring='f1_weighted', cv=cv,
                   n_jobs=-1, refit=True)


# define dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
y = y.values.ravel()

# Run the GridSearchCV class
clf.fit(X, y)

# Print the best set of hyperparameters
print(clf.best_params_, clf.best_score_)

# Finalize Model
clf = RandomForestClassifier(class_weight='balanced',
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=0.001,
                             n_estimators=75)
# define dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=X.columns)

target_names = ['Control', 'Patient']
# print(classification_report(y_test, y_pred, target_names=target_names))


class UI(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setWindowTitle("Dengue Detection System")
        self.setContentsMargins(50, 50, 50, 50)

        layout = QGridLayout()
        self.setLayout(layout)

        self.label1 = QLabel("Pregnencies : ")
        layout.addWidget(self.label1, 0, 0)

        self.label2 = QLabel("Blood Pressure : ")
        layout.addWidget(self.label2, 1, 0)

        self.label3 = QLabel("Glucose : ")
        layout.addWidget(self.label3, 2, 0)

        self.label4 = QLabel("Skin Thickness : ")
        layout.addWidget(self.label4, 3, 0)

        self.label5 = QLabel("Insulin level : ")
        layout.addWidget(self.label5, 4, 0)

        self.label6 = QLabel("BMI : ")
        layout.addWidget(self.label6, 5, 0)

        self.label7 = QLabel("Diabetes Pedigree Function : ")
        layout.addWidget(self.label7, 6, 0)

        self.label8 = QLabel("Age : ")
        layout.addWidget(self.label8, 7, 0)

        self.label9 = QLabel("Prediction : ")
        layout.addWidget(self.label9, 9, 0)

        self.input1 = QLineEdit()  # pregnencies
        layout.addWidget(self.input1, 0, 1)

        self.input2 = QLineEdit()  # bp
        layout.addWidget(self.input2, 1, 1)

        self.input3 = QLineEdit()  # Glucose
        layout.addWidget(self.input3, 2, 1)

        self.input4 = QLineEdit()  # SkinThickness
        layout.addWidget(self.input4, 3, 1)

        self.input5 = QLineEdit()  # insulin level
        layout.addWidget(self.input5, 4, 1)

        self.input6 = QLineEdit()  # bmi
        layout.addWidget(self.input6, 5, 1)

        self.input7 = QLineEdit()  # Diabetes Pedigree function
        layout.addWidget(self.input7, 6, 1)

        self.input8 = QLineEdit()  # Age
        layout.addWidget(self.input8, 7, 1)

        # self.outputbox = QLabel("Prediction ")
        # layout.addWidget(self.outputbox, 8, 0)

        self.displaybox = QLabel()
        layout.addWidget(self.displaybox, 9, 1)

        button = QPushButton("Submit")
        button.setFixedWidth(50)
        button.clicked.connect(self.disp)
        layout.addWidget(button, 8, 1, Qt.AlignmentFlag.AlignRight)

    def disp(self):
        preg = self.input1.text()
        bp = self.input2.text()
        gluc = self.input3.text()
        skin_thick = self.input4.text()
        insu = self.input5.text()
        bmi = self.input6.text()
        DPF = self.input7.text()
        AGE = self.input8.text()

        d = {'Pregnancies': int(preg), 'Glucose': int(bp), 'BloodPressure': int(gluc), 'SkinThickness': int(
            skin_thick), 'Insulin': int(insu), 'BMI': int(bmi), 'DiabetesPedigreeFunction': int(DPF), 'Age': int(AGE)}
        e = pd.DataFrame(data=d, index=[1])

        z = clf.predict(e)
        if z == 1:

            self.displaybox.setText("Diabetic")
            # Diabetic

        else:
            self.displaybox.setText("Not Diabetic")


app = QApplication(sys.argv)
window = UI()
window.show()
sys.exit(app.exec())
