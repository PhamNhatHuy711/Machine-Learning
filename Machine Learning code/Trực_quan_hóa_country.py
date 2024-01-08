import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dữ liệu từ file Excel
df = pd.read_excel('c:\\Users\\LEGION\\Downloads\\Country-data.xlsx')
# Hiển thị dữ liệu
print(df)
# Plotting a bar chart for the 'child_mort' column
plt.figure(figsize=(10,8))
plt.bar(df['health'], df['child_mort'], color='skyblue')
plt.title('Child Mortality Rate by Country')
plt.xlabel('Country')
plt.ylabel('Child Mortality Rate (per 1000)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Since the DataFrame has multiple numerical columns, we need to select one to visualize in a pie chart.
# Here we will use 'health' column as an example.

# Taking the sum of the health column for all countries to get a total
total_health = df['health'].sum()
# Calculating the percentage of health expenditure for each country
df['health_percentage'] = df['health'] / total_health * 100

# Plotting a pie chart for the 'health_percentage' column
plt.figure(figsize=(8,8))
plt.pie(df['health_percentage'], labels=df['country'], autopct='%1.1f%%', startangle=140)
plt.title('Health Expenditure by Country')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Plotting a line chart for the 'gdpp' column
plt.figure(figsize=(10,8))
plt.plot(df['country'], df['gdpp'], marker='o', color='green')
plt.title('GDP per Capita by Country')
plt.xlabel('Country')
plt.ylabel('GDP per Capita (USD)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()