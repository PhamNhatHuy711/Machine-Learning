import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("c:\\Users\\LEGION\\Downloads\\ds_salaries_1.xlsx")
df = pd.DataFrame(data)
print(df)
X = df[['experience_year','working_hours']][:300]
y = df['salary_in_usd'][:300]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Xác định feature (đặc trưng) và target (mục tiêu)



# Khởi tạo và huấn luyện mô hình Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = linear_model.predict(X_test)

# Đánh giá hiệu suất của mô hình
mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print("Linear Regression:")
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')

# Trực quan hóa dữ liệu thực tế và dự đoán
# Số lượng điểm bạn muốn giới hạn
num_points_to_plot = 7

# Lấy mẫu ngẫu nhiên từ dự đoán
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
random_indices = np.random.choice(len(y_pred_linear), size=num_points_to_plot, replace=False)
y_pred_sampled = y_pred_linear[random_indices]
X_test_sampled = X_test_np[random_indices]

# Trực quan hóa dữ liệu huấn luyện
plt.scatter(X_test['experience_year'], y_test, color='blue', label='Actual')
plt.scatter(X_test_sampled[:, 0], y_pred_sampled, color='red', label='Predicted')
plt.xlabel('Số năm làm việc',fontsize=16)
plt.ylabel('Lương', fontsize=16)
plt.legend()
plt.title('Mức lương thực tế và dự đoán', fontsize=20)
plt.plot(X_test['experience_year'], linear_model.coef_[0] * X_test['experience_year'] + linear_model.intercept_, color='green', label='Regression Line')
plt.legend()
plt.show()


# For simplicity, let's classify salaries into two categories based on the median
# This is a simple way to create a binary classification problem
median_salary = data['salary_in_usd'].median()
data['salary_class'] = (data['salary_in_usd'] >= median_salary).astype(int)

# Selecting features and target
features = ['experience_year', 'working_hours', 'remote_ratio']
target = 'salary_class'

# Preparing the features (X) and target (y)
X = data[features].fillna(0)  # Filling missing values with 0
y = data[target]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = LazyClassifier()
# models = clf.fit(X_train, X_test, y_train, y_test)
# # In ra danh sách các mô hình và điểm số đánh giá
# print(models)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fitting the model
knn.fit(X_train_scaled, y_train)

# Predicting the test set results
y_pred = knn.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define the plot
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)

# Add labels to the plot
class_names = ['Below Median', 'Above Median']
tick_marks = np.arange(len(class_names)) + 0.5
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)

plt.xlabel('Nhãn dự đoán',fontsize=16)
plt.ylabel('Nhãn thực tế',fontsize=16)
plt.title('Ma trận nhầm lẫn cho thuật toán phân loại KNN',fontsize=20)
plt.show()
#SVC
svc = SVC(kernel='linear', random_state=42)

# Fitting the model
svc.fit(X_train_scaled, y_train)

# Predicting the test set results
y_pred_svc = svc.predict(X_test_scaled)

# Evaluating the model
accuracy_svc = accuracy_score(y_test, y_pred_svc)
classification_rep_svc = classification_report(y_test, y_pred_svc)

# Compute confusion matrix for SVC
cm_svc = confusion_matrix(y_test, y_pred_svc)

# Plotting confusion matrix for SVC
plt.figure(figsize=(7, 7))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)

# Add labels to the plot
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)

plt.xlabel('Nhãn dự đoán',fontsize=16)
plt.ylabel('Nhãn thực tế',fontsize=16)
plt.title('Ma trận nhầm lẫn cho thuật toán phân loại SVC',fontsize=20)
plt.show()
#Logisic
# Initializing the Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)

# Fitting the model
log_reg.fit(X_train_scaled, y_train)

# Predicting the test set results
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluating the model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
classification_rep_log_reg = classification_report(y_test, y_pred_log_reg)

# Compute confusion matrix for Logistic Regression
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Plotting confusion matrix for Logistic Regression
plt.figure(figsize=(7, 7))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)

# Add labels to the plot
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)

plt.xlabel('Nhãn dự đoán',fontsize=16)
plt.ylabel('Nhãn thực tế',fontsize=16)
plt.title('Ma trận nhầm lẫn cho thuật toán phân loại hồi quy logistic',fontsize=20)
plt.show()


# Initialize the AdaBoostClassifier
ada_boost = AdaBoostClassifier(random_state=42)

# Fit the model
ada_boost.fit(X_train_scaled, y_train)

# Predict the test set results
y_pred_ada_boost = ada_boost.predict(X_test_scaled)

# Evaluating the model
accuracy_ada_boost = accuracy_score(y_test, y_pred_ada_boost)
classification_rep_ada_boost = classification_report(y_test, y_pred_ada_boost)

accuracy_ada_boost, classification_rep_ada_boost
# Compute confusion matrix for AdaBoostClassifier
cm_ada_boost = confusion_matrix(y_test, y_pred_ada_boost)

# Plotting confusion matrix for AdaBoostClassifier
plt.figure(figsize=(7, 7))
sns.heatmap(cm_ada_boost, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)

# Add labels to the plot
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)

plt.xlabel('Nhãn dự đoán',fontsize=16)
plt.ylabel('Nhãn thực tế',fontsize=16)
plt.title('Ma trận nhầm lẫn cho thuật toán phân loại AdaBoost',fontsize=20)
plt.show()
# Visualization of the accuracies of different models
models = [knn, svc, log_reg, ada_boost]
model_names = ['KNN', 'SVM', 'Logistic Regression', 'AdaBoostClassifier']
accuracies = []

for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Combine model names and their accuracies
model_accuracies = dict(zip(model_names, accuracies))
model_accuracies
print(model_accuracies)
# Convert the accuracies to a pandas DataFrame for easier plotting
accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=accuracy_df, color='skyblue')

# Adding labels and title
plt.xlabel('Accuracy',fontsize=16)
plt.ylabel('Classification Model',fontsize=16)
plt.title('Biểu đồ so sánh accuracy giữa các thuật toán',fontsize=20)
plt.xlim(0, 1)  # Setting x-axis limit for better readability

# Show the plot
plt.show()
