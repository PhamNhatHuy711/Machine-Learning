import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import random
import seaborn as sns
data = pd.read_excel("c:\\Users\\LEGION\\Downloads\\filedulieu2.xlsx")
df = pd.DataFrame(data)
print(df)

plt.figure(figsize=(10, 6))  # Điều chỉnh kích thước biểu đồ nếu cần thiết
plt.scatter(df['experience_year'].sample(n=200, random_state=32), df['salary_in_usd'].sample(n=200, random_state=32), alpha=0.5)  # Tạo scatter plot
plt.title('Mối tương quan giữa Experience Year và Salary in USD', fontsize=20)
plt.xlabel('Experience Year', fontsize=16)
plt.ylabel('Salary in USD', fontsize=16)
plt.grid(True)  # Hiển thị lưới
plt.show()
# # Tính ma trận tương quan
correlation_matrix = df[['experience_year', 'salary_in_usd','working_hours']].corr()
data = np.array([df['salary_in_usd'], df['experience_year'], df['working_hours']])
corr_matrix = np.corrcoef(data).round(decimals=2)
corr_matrix
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('salary_in_usd', 'experience_year', 'working_hours'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('salary_in_usd', 'experience_year', 'working_hours'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, corr_matrix[i, j], ha='center', va='center',
        color='r')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.show()