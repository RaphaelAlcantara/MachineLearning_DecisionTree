import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?', -99999, inplace=True)

X = df.drop(['mitoses'], axis=1)
y = df['mitoses']

y = y.map(lambda x: 1 if x == 1 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

feature_names = X.columns.tolist()

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, rounded=True, class_names=["Sem Mitose", "Com Mitose"], feature_names=feature_names)
plt.show()
