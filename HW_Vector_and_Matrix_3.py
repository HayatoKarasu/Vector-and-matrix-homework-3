print("1) Сингулярное разложнение - это разделение целой матрицы на части")
print("для привидения ее к удобству вычисления.")
print("Используется для работы с матрицами в машинном обучении.")

print("2) Сингулярные числа - элемены сигма матрицы расположенные по диагонали с левого верхнего угла")
print("в правый нижний угол, а столбцы по краям называются левый сингулярный вектор")
print("и правый сингулярный вектор.")

print("3) Метод главных компонент - способ уменьшить размерность множества данных")
print("при наименьших потерях информации.")

import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1, 11) * 100
y = 2 * x + np.random.randn(10)*5
X = np.vstack((x, y))
print(X)

plt.scatter(X[0], X[1])
plt.show()

Xcentered = (X[0] - x.mean(), X[1] - y.mean())
m = (x.mean(), y.mean())
print(Xcentered)
print("Mean vector: ", m)

plt.scatter(Xcentered[0], Xcentered[1])
plt.show()

covmat = np.cov(Xcentered)
print(covmat, "\n")
print("Variance of X: ", np.cov(Xcentered)[0,0])
print("Variance of Y: ", np.cov(Xcentered)[1,1])
print("Covariance X and Y: ", np.cov(Xcentered)[0,1])

_, vecs =  np.linalg.eig(covmat)
v = -vecs[:,1]
Xnew = np.dot(v, Xcentered)
print(Xnew)

print(v, v.shape)

print(Xcentered)

print(vecs[-1])

print(m)

n = 5 #номер элемента случайной величины
Xrestored = np.dot(Xnew[n],v) + m
print("Restored: ", Xrestored)
print("Original: ", X[:,n])

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
XPCAreduced = pca.fit_transform(np.transpose(X))

for xn, x_pca in zip(Xnew, XPCAreduced):
    print(xn, '-', x_pca)