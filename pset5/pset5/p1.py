import numpy as np
import numpy.linalg
alpha = 0.99


psi = np.array([
    [alpha, 1 - alpha],
    [1 - alpha, alpha]
])

phi = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])

d = np.array([0, 1]).T
e = np.array([0, 1]).T

m_da = np.matmul(phi, d)
m_ec = np.matmul(phi, e)
m_cb = np.matmul(psi, m_ec)
m_ba = np.matmul(psi, m_cb)

a = np.multiply(m_ba, m_da)
print(a)
print(a / np.sum(a))
