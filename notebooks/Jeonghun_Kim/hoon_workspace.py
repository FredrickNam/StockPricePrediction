import math
import matplotlib.pyplot as plt

#1. 활성화함수와 미분
def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1-s)

# 2. 입력값, 정답, 초기 파라미터
x = 2.0
y = 1.0

w1 = 0.5
b1 = 0.0
w2 = -0.3
b2 = 0.0

learning_rate = 0.1
epochs = 100

loss_history=[] 

# 3. 순전파 (forward propagation)
'''
z1 = w1*x+b1
a1 = sigmoid(z1)
z2 = w2*a1+b2
y_hat = z2
loss = 0.5*(y_hat-y)**2
'''
for epoch in range(epochs):
    z1 = w1*x+b1
    a1 = sigmoid(z1)
    z2 = w2*a1+b2
    y_hat = z2
    loss = 0.5*(y_hat-y)**2

    loss_history.append(loss)

    dL_dyhat = y_hat - y

    # y_hat = z2 이므로
    dL_dz2 = dL_dyhat

    # z2 = w2 * a1 + b2
    dL_dw2 = dL_dz2 * a1
    dL_db2 = dL_dz2
    dL_da1 = dL_dz2 * w2

    # a1 = sigmoid(z1)
    dL_dz1 = dL_da1 * sigmoid_derivative(z1)

    # z1 = w1 * x + b1
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    # 5. 경사하강법 업데이트
    w1 = w1 - learning_rate * dL_dw1
    b1 = b1 - learning_rate * dL_db1
    w2 = w2 - learning_rate * dL_dw2
    b2 = b2 - learning_rate * dL_db2
'''
print("=== Forward ===")
print(f"z1 = {z1:.6f}")
print(f"a1 = {a1:.6f}")
print(f"z2 = {z2:.6f}")
print(f"y_hat = {y_hat:.6f}")
print(f"loss = {loss:.6f}")
'''
'''
# 4. 역전파 (backpropagation)
# 출력층 오차
dL_dyhat = y_hat - y

# y_hat = z2 이므로
dL_dz2 = dL_dyhat

# z2 = w2 * a1 + b2
dL_dw2 = dL_dz2 * a1
dL_db2 = dL_dz2
dL_da1 = dL_dz2 * w2

# a1 = sigmoid(z1)
dL_dz1 = dL_da1 * sigmoid_derivative(z1)

# z1 = w1 * x + b1
dL_dw1 = dL_dz1 * x
dL_db1 = dL_dz1
'''
'''
print("\n=== Backward ===")
print(f"dL/dw2 = {dL_dw2:.6f}")
print(f"dL/db2 = {dL_db2:.6f}")
print(f"dL/dw1 = {dL_dw1:.6f}")
print(f"dL/db1 = {dL_db1:.6f}")
'''
'''
# 5. 경사하강법 업데이트
w1 = w1 - learning_rate * dL_dw1
b1 = b1 - learning_rate * dL_db1
w2 = w2 - learning_rate * dL_dw2
b2 = b2 - learning_rate * dL_db2
'''
'''
print("\n=== Updated Parameters ===")
print(f"w1 = {w1:.6f}")
print(f"b1 = {b1:.6f}")
print(f"w2 = {w2:.6f}")
print(f"b2 = {b2:.6f}")
'''
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.grid(True)
plt.show()
