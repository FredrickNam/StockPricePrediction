import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

w = 1.0
b = 3.0

learning_rate = 0.01
epochs = 1000

n = len(x)
loss_history = []

y_hat=[0]*n
for epoch in range(epochs):
    tmp1=0
    tmp2=0
    tmp3=0
    for i in range(n):
        y_hat[i]=w*x[i]+b

    for i in range(n):
        tmp1 += (y_hat[i]-y[i])**2

    loss_history.append(tmp1/n)

    for i in range(n):
        tmp2 += (y_hat[i]-y[i])*x[i]
    dL_dw = 2*tmp2/n

    for i in range(n):
        tmp3 += (y_hat[i]-y[i])
    dL_db = 2*tmp3/n

    #print(w, b)
    w = w-learning_rate*dL_dw
    b = b-learning_rate*dL_db

plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.grid(True)
plt.show()


    

    
    


