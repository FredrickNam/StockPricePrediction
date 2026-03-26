import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 0]  
y = iris.target      

y = (y == 0).astype(int) 

w = 0.5   
b = 0.5   
lr = 0.01 
epochs = 100 

print(f"학습 시작 전 가중치: w={w:.2f}, b={b:.2f}")

for epoch in range(epochs):
    # 예측값 계산 (Y = wX + b)
    prediction = w * X + b
    
    # 오차(Loss) 계산 - 평균 제곱 오차(MSE)
    error = prediction - y
    loss = np.mean(error**2)
    
    # 미분을 통한 기울기(Gradient) 계산
    w_gradient = 2 * np.mean(error * X)
    b_gradient = 2 * np.mean(error)
    
    # 가중치 업데이트 (경사하강법 핵심!)
    w = w - lr * w_gradient
    b = b - lr * b_gradient
    
    # 10번마다 손실값 출력
    if (epoch + 1) % 10 == 0:
        print(f"에포크 {epoch+1}: Loss = {loss:.4f}, w = {w:.2f}, b = {b:.2f}")

print("-" * 30)
print(f"학습 완료 후 가중치: w={w:.2f}, b={b:.2f}")
