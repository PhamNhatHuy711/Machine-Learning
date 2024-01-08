
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import missingno as msno

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import matplotlib
matplotlib.rcParams["figure.figsize"] = (16, 6)
import warnings
warnings.filterwarnings('ignore')



df = pd.read_excel('c:\\Users\\LEGION\\Downloads\\ds_salaries_1.xlsx')
print(df.head(5))
df.info()
#Tìm các giá trị trùng lặp
print(df.duplicated().sum())
# Xác định các hàng trùng lặp
duplicated_rows = df.duplicated()
# Tạo một DataFrame mới để lưu số thứ tự của hàng và xác định liệu chúng có bị trùng lặp không
duplicates_data = pd.DataFrame({
'Row': range(1, len(df) + 1),
'Duplicated': duplicated_rows
})

# Lọc ra chỉ các hàng đã trùng lặp
duplicates_data = duplicates_data[duplicates_data['Duplicated']]
# Vẽ chúng dưới dạng đường thẳng màu đen
plt.figure(figsize=(10, 6))
plt.vlines(x=duplicates_data['Row'], ymin=0, ymax=1, color='black')
plt.gca().invert_xaxis() # Đảo ngược trục x
plt.title("Chỉ số của các hàng trùng lặp ")
plt.show()
# Kiểm tra các hàng có giá trị trùng lặp
duplicate_rows = df[df.duplicated()]

# In ra các hàng có giá trị trùng lặp
print("Các hàng có giá trị trùng lặp:")
print(duplicate_rows)
#Tìm Missing values
print(df.isna().sum())
msno.bar(df)
plt.title('Phân phối các Missing Values', fontsize=16,
fontstyle='oblique')
plt.show()
# Xử lý dữ liệu nhiễu (Dùng IQR)
def find_outliers(column):
    global q_list
    q_list = []
    
    # Chuyển đổi giá trị của cột thành số, bỏ qua giá trị không thể chuyển đổi
    numeric_values = pd.to_numeric(column, errors='coerce')
    # Lấy chỉ mục của các giá trị không thể chuyển đổi thành số
    non_numeric_indices = pd.to_numeric(df['salary_in_usd'], errors='coerce').isna()
    non_numeric_values = df.loc[non_numeric_indices, 'salary_in_usd']
    df['salary_in_usd'] = pd.to_numeric(df['salary_in_usd'], errors='coerce')
    
    # Sắp xếp DataFrame theo cột số
    sorted_df = numeric_values.sort_values()
    
    for q, p in {"Q1": 25, "Q2": 50, "Q3": 75}.items():
        # Calculate Q1, Q2, Q3 and IQR.
        Q = np.percentile(sorted_df, p, interpolation='midpoint')
        q_list.append(Q)
        print("Checking...", q)
        time.sleep(2)
        print("{}: {} percentile of the {} values is,".format(q, p, column.name), Q)
        
    global Q1, Q2, Q3
    Q1 = q_list[0]
    Q2 = q_list[1]
    Q3 = q_list[2]
    IQR = Q3 - Q1
    print("Interquartile range is", IQR)

    #  Tìm giới hạn dưới và giới hạn trên là Q1 – 1,5 IQR và Q3 + 1,5 IQR, tương ứng
    global low_lim, up_lim
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    time.sleep(1)
    print(" ")
    print("Checking limits")
    time.sleep(2)
    print("low_limit is", low_lim)
    print("up_limit is", up_lim)

    time.sleep(1)
    # Find outliers in the dataset
    outliers = sorted_df[(sorted_df > up_lim) | (sorted_df < low_lim)]
    
    print("\nOutliers in the dataset are being added to the list. Please wait!")
    time.sleep(3)
    print("\nOutliers in the dataset are", outliers)
# Gọi hàm find_outliers với cột 'salary_in_usd'
find_outliers(df['salary_in_usd'])
# Trựa quan hóa
def plot_outliers(df):
    f, ax = plt.subplots(figsize=(22,5))
    ax.ticklabel_format(style='plain', axis='both')
    outliers = sns.boxplot(ax=ax, x=df, palette="Paired")
    plt.axvspan(xmin = low_lim, xmax = df.min(), alpha=0.3,
    color='red')
    plt.axvspan(xmin = up_lim, xmax = df.max(), alpha=0.3,
    color='red')
    plt.show()
plot_outliers(df['salary_in_usd'])
clean_data = df[(df['salary_in_usd'] < up_lim) & (df['salary_in_usd'] > low_lim)]
print("Minimum salary in USD:{}".format(clean_data['salary_in_usd'].min()))
print("Maximum salary in USD:{}".format(clean_data['salary_in_usd'].max()))
print(clean_data.head())
#Trực quan hóa 
def plot_outliers(clean_data):
# Paint red outlier areas on the boxplot
    f, ax = plt.subplots(figsize=(22,5))
    ax.ticklabel_format(style='plain', axis='both')
    outliers = sns.boxplot(ax=ax, x=clean_data, palette="Paired")
    plt.axvspan(xmin = low_lim, xmax = clean_data.min(), alpha=0.3,
    color='red')
    plt.axvspan(xmin = up_lim, xmax = clean_data.max(), alpha=0.3,
    color='red')
    plt.show()
plot_outliers(clean_data['salary_in_usd'])
# #Chuẩn hóa dữ liẹu
def summarize_dataframe(clean_data):

    # In ra số lượng bản ghi và cột
    print(f"\nCó {len(clean_data)} bản ghi và {len(clean_data.columns)} thuộc tính/cột")

    # Xác định các thuộc tính số học
    numeric_features = clean_data.select_dtypes(include=[np.number]).columns.tolist()

    # Xác định các thuộc tính phân loại
    categorical_features = clean_data.select_dtypes(exclude=[np.number]).columns.tolist()

    # In ra số lượng thuộc tính số học và từng thuộc tính
    print(f"\nCó {len(numeric_features)} thuộc tính numeric: \n")
    for i, feature in enumerate(numeric_features, 1):
        print(f"{i}. {feature}\n")

        # In ra số lượng thuộc tính phân loại và từng thuộc tính
    print(f"\nCó {len(categorical_features)} thuộc tính categorical:\n")
    for i, feature in enumerate(categorical_features, 1):
            print(f"{i}. {feature} \n")
            
summarize_dataframe(clean_data)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col1 = ['employee_residence', 'company_location', 'company_size', 'employment_type']
clean_data[col1] = clean_data[col1].apply(LabelEncoder().fit_transform)

# Tạo một đối tượng MinMaxScaler
scaler = MinMaxScaler()

# Chọn các cột cần chuẩn hóa
columns_to_scale = ['remote_ratio', 'salary_in_usd', 'salary', 'working_hours', 'experience_year']

# Chuẩn hóa Min-Max
clean_data[columns_to_scale] = scaler.fit_transform(clean_data[columns_to_scale])
clean_data.to_excel('c:\\Users\\LEGION\\Downloads\\ds_salaries_preprocessing.xlsx', index=True)
print(clean_data)