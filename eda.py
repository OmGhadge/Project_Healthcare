import pandas as pd 
# import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import numpy as np

df=pd.read_csv("Data/dataset.csv")
print(df.head())

# #check for null values
# print(df.isnull().sum())

# #check for corelation
# print(df.corr())

# # #encode categorical values
# # df=pd.get_dummies(df,drop_first=True)
# #identify the outliers with boxplots 
# # List of column names for which you want to create boxplots
# selected_columns = ['age', 'avg_glucose_level', 'bmi']

# # Create boxplots for the selected columns
# plt.figure(figsize=(10, 6))  # Set the figure size

# # Loop through each selected column and create a boxplot
# for i, column in enumerate(selected_columns):
#     plt.subplot(1, len(selected_columns), i + 1)
#     plt.boxplot(df_copy[column])
#     plt.title(column)

# plt.tight_layout()
# plt.show()

df_copy = df.copy()

# # Fill NaN values in 'bmi' column with the mean for visualization
# df_copy['bmi'].fillna(df_copy['bmi'].mean(), inplace=True)

# # Assuming 'df_copy' is a copy of your original DataFrame 'df'
# Q1_glucose = df_copy['avg_glucose_level'].quantile(0.25)
# Q3_glucose = df_copy['avg_glucose_level'].quantile(0.75)
# IQR_glucose = Q3_glucose - Q1_glucose

# Q1_bmi = df_copy['bmi'].quantile(0.25)
# Q3_bmi = df_copy['bmi'].quantile(0.75)
# IQR_bmi = Q3_bmi - Q1_bmi

# # Calculate upper bounds for outliers
# upper_bound_glucose = Q3_glucose + 1.5 * IQR_glucose
# upper_bound_bmi = Q3_bmi + 1.5 * IQR_bmi

# # Retrieve outliers in 'avg_glucose_level' and 'bmi' columns from df_copy
# outliers_glucose = df_copy[df_copy['avg_glucose_level'] > upper_bound_glucose]['avg_glucose_level']
# outliers_bmi = df_copy[df_copy['bmi'] > upper_bound_bmi]['bmi']

# print("Outliers in 'avg_glucose_level':")
# print(outliers_glucose)

# print("\nOutliers in 'bmi':")
# print(outliers_bmi)

# # describe df_copy
# print(df_copy.describe())
# # Set the style and context of the plot
# sns.set_style("whitegrid")
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# # Create the figure and the axes (subplots) on it
# fig, ax = plt.subplots(figsize=(10, 8))

# # Create the countplot
# sns.countplot(x='stroke', data=df, palette='viridis', ax=ax)

# # Set the title and labels
# ax.set_title('Stroke Count', fontsize=20)
# ax.set_xlabel('Stroke', fontsize=15)
# ax.set_ylabel('Number of Occurrences', fontsize=15)

# # Show the plot
# plt.show()

# #get all the columns
# print(df.columns)
# #encoding the categorical variables

# df_copy['gender'] = df_copy['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
# df_copy['Residence_type'] = df_copy['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
# df_copy['work_type'] = df_copy['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
# df_copy.head()

# DT_bmi_pipe=Pipeline(steps=[('scale',StandardScaler()),('lr',DecisionTreeRegressor(random_state=42))])
            
# #copy relevant columns for imputation
# X=df_copy[['age','gender','bmi']].copy()
# X.gender=X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype('np.uint8')

# #seperate missing values and non missing values
# Missing=X[X.bmi.isna()]
# X=X[~X.bmi.isna()]
# Y=X.pop('bmi')

# #fit the pipeline on non-missing data
# DT_bmi_pipe.fit(X,Y)

# #predict the missing 'bmi' values
# predicted_bmi=pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]) ,index=Missing.index)

# #Fill missing 'bmi'values in the original dataframe
# df_copy.loc[Missing.index,'bmi']=predicted_bmi



# # Show the plot
# plt.show()

# # get all the columns
# print(df.columns)

# encoding the categorical variables
# One-hot encoding for 'gender', 'Residence_type', 'work_type', and 'smoking_status'
# List of columns for one-hot encoding
columns_to_encode = ['gender','work_type', 'smoking_status']

#perform one hot encoding on column_to_encode and make drop_first=True
df_encoded = pd.get_dummies(df_copy, columns=columns_to_encode, drop_first=True)


# Encoding 'ever_married' as binary (Yes:1, No:0)
df_encoded['ever_married'] = df_encoded['ever_married'].replace({'Yes': 1, 'No': 0}).astype(np.uint8)
df_copy['Residence_type'] = df_copy['Residence_type'].replace({'Rural': 0, 'Urban': 1}).astype(np.uint8)
print(df_encoded.columns)

df_encoded.drop(columns=['id','ever_married','Residence_type','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes'], inplace=True)

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X=df_encoded.drop(columns=['stroke'])
y=df_encoded['stroke']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#check data types of all columns
print(df_encoded.dtypes)

DT_bmi_pipe = Pipeline(steps=[('scale',StandardScaler()),('lr',DecisionTreeRegressor(random_state=42))])

# copy relevant columns for imputation
Xb = X_train[['age', 'gender_Male', 'gender_Other', 'bmi']].copy()

# separate missing values and non missing values
Missing = Xb[Xb.bmi.isna()]
Xb = Xb[~Xb.bmi.isna()]
Yb = Xb.pop('bmi')

# fit the pipeline on non-missing data
DT_bmi_pipe.fit(Xb,Yb)

# predict the missing 'bmi' values
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age', 'gender_Male', 'gender_Other']]), index=Missing.index)
X_train.loc[Missing.index, 'bmi'] = predicted_bmi
print(X_train.head())



# # Random upsampling
# from sklearn.utils import resample

# # Separating majority and minority classes
# majority_class = df_encoded[df_encoded['stroke'] == 0]
# minority_class = df_encoded[df_encoded['stroke'] == 1]

# # Upsample minority class
# minority_upsampled = resample(minority_class,
#                               replace=True,  # Sample with replacement
#                               n_samples=len(majority_class),  # Match number of majority class
#                               random_state=42)  # Set random state for reproducibility

# # Combine upsampled minority class with original majority class
# df_upsampled = pd.concat([majority_class, minority_upsampled])
 
#random upsampling using imblearn
from imblearn.over_sampling import RandomOverSampler 


# Assuming 'df_encoded' contains your dataset with classes in 'stroke'

# Separating features and target variable
Xc = X_train.drop(columns=['stroke'])  # Features
yc = X_train['stroke']  # Target variable
Xc.columns

# Creating the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Resampling the dataset
X_resampled, y_resampled = ros.fit_resample(Xc, yc)
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=Xc.columns), pd.Series(y_resampled, name='stroke')], axis=1)
df_resampled

#visualization before upsampling
# Set the style and context of the plot
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Create the figure and the axes (subplots) on it
fig, ax = plt.subplots(figsize=(10, 8))

# Create the countplot
sns.countplot(x='stroke', data=X_train, palette='viridis', ax=ax)

# Set the title and labels
ax.set_title('Stroke Count', fontsize=20)
ax.set_xlabel('Stroke', fontsize=15)
ax.set_ylabel('Number of Occurrences', fontsize=15)

# Show the plot
plt.show()

#visualization after upsampling
# Set the style and context of the plot
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Create the figure and the axes (subplots) on it
fig, ax = plt.subplots(figsize=(10, 8))

# Create the countplot
sns.countplot(x='stroke', data=df_resampled, palette='viridis', ax=ax)

# Set the title and labels
ax.set_title('Stroke Count', fontsize=20)
ax.set_xlabel('Stroke', fontsize=15)
ax.set_ylabel('Number of Occurrences', fontsize=15)

# Show the plot
plt.show()

#Decision function before and after upsampling

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X = df_encoded[['age', 'avg_glucose_level']]
y = df_encoded['stroke']
# Assuming 'X' contains your original features and 'y' contains the original target variable
# Assuming 'X_resampled' contains the resampled features and 'y_resampled' contains the resampled target variable
clf_visualize = RandomForestClassifier(random_state=42)
# Creating a RandomForestClassifier (or any classifier of your choice)
# clf_before = RandomForestClassifier(random_state=42)
# clf_after = RandomForestClassifier(random_state=42)

clf_visualize.fit(X, y)
# Visualize decision boundary before oversampling
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Decision Function before Oversampling')

# Create a meshgrid for decision boundary visualization before oversampling
xx, yy = np.meshgrid(np.linspace(X['age'].min(), X['age'].max(), 100),
                     np.linspace(X['avg_glucose_level'].min(), X['avg_glucose_level'].max(), 100))
X_mesh = np.c_[xx.ravel(), yy.ravel()]

Z = clf_visualize.predict(X_mesh).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])

plt.scatter(X[y == 0]['age'], X[y == 0]['avg_glucose_level'], label='Class 0')
plt.scatter(X[y == 1]['age'], X[y == 1]['avg_glucose_level'], label='Class 1')

plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.legend()

# Train the classifier on the oversampled data
clf_visualize.fit(X_resampled, y_resampled)

# Visualize decision boundary after oversampling
plt.subplot(1, 2, 2)
plt.title('Decision Function after Random Oversampling')

xx_resampled, yy_resampled = np.meshgrid(np.linspace(X_resampled['age'].min(), X_resampled['age'].max(), 100),
                                         np.linspace(X_resampled['avg_glucose_level'].min(), X_resampled['avg_glucose_level'].max(), 100))
X_resampled_mesh = np.c_[xx_resampled.ravel(), yy_resampled.ravel()]

Z_resampled = clf_visualize.predict(X_resampled_mesh).reshape(xx_resampled.shape)

plt.contourf(xx_resampled, yy_resampled, Z_resampled, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])

plt.scatter(X_resampled[y_resampled == 0]['age'], X_resampled[y_resampled == 0]['avg_glucose_level'], label='Class 0')
plt.scatter(X_resampled[y_resampled == 1]['age'], X_resampled[y_resampled == 1]['avg_glucose_level'], label='Class 1')

plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.legend()

plt.tight_layout()
plt.show()
# # Fitting the classifiers on the original and resampled data
# clf_before.fit(X, y)
# clf_after.fit(X_resampled, y_resampled)

# Creating subplots for before and after oversampling decision function visualization
# # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# plt.figure(figsize=(12, 5))

# # Plotting decision function before oversampling
# plt.subplot(1, 2, 1)
# plt.title('Decision Function before Random Oversampling')

# # Assuming you have 2D features for visualization (replace with actual features)
# plt.scatter(X[y == 0]['age'], X[y == 0]['avg_glucose_level'], label='Class 0')
# plt.scatter(X[y == 1]['age'], X[y == 1]['avg_glucose_level'], label='Class 1')

# # Create a meshgrid for decision boundary visualization before oversampling
# xx, yy = np.meshgrid(np.linspace(X['age'].min(), X['age'].max(), 100),
#                      np.linspace(X['avg_glucose_level'].min(), X['avg_glucose_level'].max(), 100))

# X_mesh = np.c_[xx.ravel(), yy.ravel()]  # Create mesh data with 'age' and 'avg_glucose_level'

# Z = clf_before.predict(X_mesh).reshape(xx.shape)  # Predict using the entire mesh data

# plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])

# plt.scatter(X[y == 0]['age'], X[y == 0]['avg_glucose_level'], label='Class 0')
# plt.scatter(X[y == 1]['age'], X[y == 1]['avg_glucose_level'], label='Class 1')

# plt.xlabel('Age')
# plt.ylabel('Average Glucose Level')
# plt.legend()

# # Plotting decision function after oversampling
# plt.subplot(1, 2, 2)
# plt.title('Decision Function after Random Oversampling')

# # Assuming you have 2D features for visualization (replace with actual features)
# plt.scatter(X_resampled[y_resampled == 0]['age'], X_resampled[y_resampled == 0]['avg_glucose_level'], label='Class 0')
# plt.scatter(X_resampled[y_resampled == 1]['age'], X_resampled[y_resampled == 1]['avg_glucose_level'], label='Class 1')

# # Create a meshgrid for decision boundary visualization after oversampling
# xx_resampled, yy_resampled = np.meshgrid(np.linspace(X_resampled['age'].min(), X_resampled['age'].max(), 100),
#                                          np.linspace(X_resampled['avg_glucose_level'].min(), X_resampled['avg_glucose_level'].max(), 100))
# Z_resampled = clf_after.predict(np.c_[xx_resampled.ravel(), yy_resampled.ravel()]).reshape(xx_resampled.shape)

# # Plotting decision boundary
# plt.contourf(xx_resampled, yy_resampled, Z_resampled, alpha=0.3, levels=[0, 0.5, 1], colors=['blue', 'red'])

# plt.xlabel('Age')
# plt.ylabel('Average Glucose Level')
# plt.legend()

# plt.tight_layout()
# plt.show()

# we finished at getting the visual but not as given in the 
# sklearn documentation. In chatgpt similar code is given try it out 
# https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.htmldef plot_decision_function(X, y, clf, ax, title=None):
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LogisticRegression

# # Create a logistic regression classifier
# clf = LogisticRegression()

# # Fit the classifier to the training data
# clf.fit(X_train[['age', 'avg_glucose_level']], y_train)

# # Define the step size for plotting the decision boundaries
# plot_step = 0.1  # Adjusting the step size for better granularity

# # Define the range for the meshgrid based on feature ranges
# x_min, x_max = X_train['age'].min() - 1, X_train['age'].max() + 1
# y_min, y_max = X_train['avg_glucose_level'].min() - 1, X_train['avg_glucose_level'].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

# # Predict the target on the meshgrid points and reshape for contour plot
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot decision boundaries
# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, alpha=0.4, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])

# # Plot the training data points
# plt.scatter(X_train['age'], X_train['avg_glucose_level'], alpha=0.8, c=y_train, edgecolor="k")

# # Set title and labels
# plt.title("Decision Function")
# plt.xlabel('Age')
# plt.ylabel('Avg Glucose Level')

# # Show the plot
# plt.show()

    
 # SKLEARN DOCUMENTATION   
# import numpy as np

# from sklearn.datasets import make_classification


# def create_dataset(
#     n_samples=1000,
#     weights=(0.01, 0.01, 0.98),
#     n_classes=3,
#     class_sep=0.8,
#     n_clusters=1,
# ):
#     return make_classification(
#         n_samples=n_samples,
#         n_features=2,
#         n_informative=2,
#         n_redundant=0,
#         n_repeated=0,
#         n_classes=n_classes,
#         n_clusters_per_class=n_clusters,
#         weights=list(weights),
#         class_sep=class_sep,
#         random_state=0,
#     )
# def plot_decision_function(X, y, clf, ax, title=None):
#     plot_step = 0.02
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(
#         np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
#     )

#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, alpha=0.4)
#     ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
#     if title is not None:
#         ax.set_title(title)
        
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import RandomOverSampler
# clf = LogisticRegression()        
# X_train.columns
# X_train_copy = X_train.copy()
# X_train.drop(columns=['hypertension','heart_disease','bmi', 'gender_Male', 'gender_Other', 'work_type_Never_worked',
#        'work_type_Private', 'work_type_Self-employed', 'work_type_children'], inplace=True)
# y_train_copy = y_train.copy()
# X_train.head()
# X_train.reset_index(drop=True, inplace=True)
# from imblearn.pipeline import make_pipeline

# X, y = create_dataset(n_samples=100, weights=(0.05, 0.25, 0.7))

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

# clf.fit(X_train, y_train)
# plot_decision_function(X_train, y_train, clf,axs[0],title="Without resampling")

# sampler = RandomOverSampler(random_state=0)
# model = make_pipeline(sampler, clf).fit(X_train, y_train)
# plot_decision_function(X_train, y_train, model,axs[1] ,f"Using {model[0].__class__.__name__}")

# fig.suptitle(f"Decision function of {clf.__class__.__name__}")
# fig.tight_layout()

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

X_train
y_train.columns
# Function to plot decision boundary
def plot_decision_boundary(X, y, clf, ax, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)

# Assuming X_train contains 'age' and 'avg_glucose_level', and y_train contains 'stroke'
# Fit the logistic regression model
clf = LogisticRegression()
clf.fit(X_train[['age', 'avg_glucose_level']], y_train)

# Visualize decision boundary before resampling
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(X_train.values, y_train.values, clf, axs[0], title="Before Resampling")

# Apply RandomOverSampler for resampling
sampler = RandomOverSampler(random_state=0)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

# Fit the model on resampled data
clf_resampled = LogisticRegression()
clf_resampled.fit(X_resampled[['age', 'avg_glucose_level']], y_resampled)

# Visualize decision boundary after resampling
plot_decision_boundary(X_resampled.values, y_resampled.values, clf_resampled, axs[1], title="After Random OverSampling")

plt.tight_layout()
plt.show()

# done with the visual now do the same for the smote and other tech

