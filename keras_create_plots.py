import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------

# read csv with semicolon as delimeter
df = pd.read_csv('files_keras/results/results2023-02-07_22-19-24.csv', sep=';')
acc = df['acc']
model_number = df['model_number']
elapsed_time = df['elapsed_time']

plt.plot(model_number, acc)
plt.show()