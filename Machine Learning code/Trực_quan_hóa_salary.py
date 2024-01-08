import pandas as pd
import numpy as np
# import country_converter as coco
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
# import plotly.graph_objects as go
# from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
# import nltk
df = pd.read_excel('c:\\Users\\LEGION\\Downloads\\ds_salaries_preprocessing.xlsx')
print(df)
top15_job_titles = df['job_title'].value_counts()[:15]
fig = px.bar(y = top15_job_titles.values, x = top15_job_titles.index, 
            text = top15_job_titles.values, title = '15 công việc phổ biến nhất')
fig.update_layout(xaxis_title="công việc", yaxis_title="Số lượng", title_x=0.5)  
fig.update_layout(title_font=dict(size=40))
fig.update_layout(xaxis_title_font=dict(size=30))
fig.update_layout(yaxis_title_font=dict(size=30))
fig.show()

specific_job_titles = ['Machine Learning Engineer', 'Data Analyst', 'Data Engineer', 'Data Scientist']
df['job_title_modified'] = df['job_title'].apply(lambda x: x if x in specific_job_titles else 'Another Job')

# Calculate the distribution for the modified job titles
job_title_distribution_modified = df['job_title_modified'].value_counts()

# Create the pie chart
plt.figure(figsize=(10, 10))
plt.pie(job_title_distribution_modified, labels=job_title_distribution_modified.index, autopct='%1.1f%%', startangle=140)
plt.title('Phân bổ chức danh công việc cụ thể và những chức danh khác',fontsize=20)
plt.show()
plt.figure(figsize=(12, 6))

sns.boxplot(data=df, x='experience_year', y='salary_in_usd')
plt.title('Phân bổ lương theo số năm kinh nghiệm',fontsize=20)
plt.xlabel('Năm kinh nghiệm', fontsize=20)
plt.ylabel('Lương', fontsize=20)
plt.show()