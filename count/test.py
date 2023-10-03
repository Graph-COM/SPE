import numpy as np



# randomly generate Laplacian
count = [0, 0]
for i in range(9999):
    print(i)
    A = np.random.randint(0, 2, size=[10, 10])
    A = np.triu(A, k=0)
    A = A + A.T
    A[np.arange(10), np.arange(10)] = A[np.arange(10), np.arange(10)] / 2
    D = np.diag(1/A.sum(axis=0))
    L1 = np.eye(10) - D**(1/2)@A@D**(1/2)

    A = np.random.randint(0, 2, size=[10, 10])
    A = np.triu(A, k=0)
    A = A + A.T
    A[np.arange(10), np.arange(10)] = A[np.arange(10), np.arange(10)] / 2
    D = np.diag(1/A.sum(axis=0))
    L2 = np.eye(10) - D**(1/2)@A@D**(1/2)

    if np.linalg.norm(L1 - L2) <= 1:
        count[0] += 1
    else:
        count[1] += 1
print(count)





# define a Gaussian process
mu = np.array([1 for i in range(10)]).reshape([-1, 1])
K = np.array([[np.exp(-np.abs(i-j)) for i in range(10)] for j in range(10)])

# ground truth
def gt(x):
    return 0.5 * x

# observation
X = np.array([0, 1, 2])
f = gt(X).reshape([-1, 1])
K_ob = np.array([[np.exp(-np.abs(i-j)) for i in X] for j in X])

# posterior
X_new = np.array([i for i in range(10)])
K_new = np.array([[np.exp(-np.abs(i-j)) for i in X_new] for j in X_new])
K_new_ob = np.array([[np.exp(-np.abs(i-j)) for i in X] for j in X_new])
mu_new = mu + K_new_ob @ np.linalg.inv(K_ob) @ f
K_new = K_new - K_new_ob @ np.linalg.inv(K_ob) @ K_new_ob.T

pass

