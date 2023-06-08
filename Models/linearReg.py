# Implementation of Linear Regression Model
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv('sampleData.csv')

# Scaling the data
data['SAT'] = data['SAT'] / 1000

# Plotting the data
# plt.scatter(data['SAT'], data['GPA'])
# plt.xlabel('SAT')
# plt.ylabel('GPA')
# plt.show()


# def lossFunction(m, b, points):
#     totalError = 0
#     for i in range(0, len(points)):
#         x = points.iloc[i].sat
#         y = points.iloc[i].gpa
#         totalError += (y - (m * x + b)) ** 2
#     return totalError / float(len(points))

def gradientDescent(mNow, bNow, points, L):
    mGrad = 0
    bGrad = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points.iloc[i].SAT
        y = points.iloc[i].GPA
        mGrad += -(2/N) * x * (y - (mNow * x + bNow))
        bGrad += -(2/N) * (y - (mNow * x + bNow))

    newM = mNow - (L * mGrad)
    newB = bNow - (L * bGrad)
    return [newM, newB]


m = 0
b = 0
L = 0.001
epochs = 1000

for i in range(epochs):
    m, b = gradientDescent(m, b, data, L)

print(m, b)

# Plotting the data
plt.scatter(data['SAT'], data['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
x = data['SAT']
plt.plot(x, m * x + b, color='red')
plt.show()
