import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?', -99999, inplace=True)

X = df.drop(['mitoses'], axis=1)
y = df['mitoses']

y = y.map(lambda x: 1 if x == 1 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Arvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_tree = clf.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Acurácia da Árvore de decisão: {accuracy_tree:.2f}")

print("Matriz de Confusão da Decision Tree:")
print(confusion_matrix(y_test, y_pred_tree))

print("Relatório de Classificação da Árvore de decisão:")
print(classification_report(y_test, y_pred_tree))

# K-Nearest Neighbors (KNN) Classifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Acurácia do KNN: {accuracy_knn:.2f}")

print("Matriz de Confusão do KNN:")
print(confusion_matrix(y_test, y_pred_knn))

print("Relatório de Classificação do KNN:")
print(classification_report(y_test, y_pred_knn))

# Plot gráfico da árvore de decisão
feature_names = X.columns.tolist()

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, rounded=True, class_names=["Sem Mitose", "Com Mitose"], feature_names=feature_names)
plt.show()
