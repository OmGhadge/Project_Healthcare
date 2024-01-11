
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv("Data/dataset.csv")
print(df.head())

df_copy = df.copy()

columns_to_encode = ['gender','work_type', 'smoking_status']

#perform one hot encoding on column_to_encode and make drop_first=True
df_encoded = pd.get_dummies(df_copy, columns=columns_to_encode, drop_first=True)

df_encoded.drop(columns=['id','ever_married','Residence_type','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes'], inplace=True)

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X=df_encoded.drop(columns=['stroke'])
y=df_encoded['stroke']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#filling nan values in train data
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
X_train.loc[Missing.index,'bmi'] = predicted_bmi
print(X_train.head())


X_train.columns
#dropping columns
X_train.drop(columns=['hypertension', 'heart_disease','gender_Male', 'gender_Other', 'work_type_Never_worked','work_type_Private', 'work_type_Self-employed', 'work_type_children'],inplace=True)
y_train.value_counts()
X_train.drop(columns=['bmi'],inplace=True)
X_train.columns

#--------------------------------------------------------------
from sklearn.cluster import MiniBatchKMeans
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
fig, axs = plt.subplots(3, 2, figsize=(12, 15))

plot_decision_boundary(X_train.values, y_train.values, clf, axs[0, 0], title="Before Resampling")

# Apply each resampling technique and visualize decision boundary
samplers = [SMOTE(random_state=0), BorderlineSMOTE(random_state=0), KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_clusters=100,n_init=1,random_state=0),random_state=0,cluster_balance_threshold=0.2),
            SVMSMOTE(random_state=0), ADASYN(random_state=0)]

titles = ["After SMOTE", "After BorderlineSMOTE", "After KMeansSMOTE", "After SVMSMOTE", "After ADASYN"]

# for i, sampler in enumerate(samplers):
#     X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
#     clf_resampled = LogisticRegression()
#     clf_resampled.fit(X_resampled[['age', 'avg_glucose_level']], y_resampled)
    
#     plot_decision_boundary(X_resampled.values, y_resampled.values, clf_resampled, axs[i+1, 0], title=titles[i])

# plt.tight_layout()
# plt.show()

# samplers = [SMOTE(random_state=0), BorderlineSMOTE(random_state=0), KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_clusters=100, n_init=1, random_state=0), random_state=0, cluster_balance_threshold=0.2),
#             SVMSMOTE(random_state=0), ADASYN(random_state=0)]

# # Adjusting subplot to visualize two resampling techniques at a time
# fig, axs = plt.subplots(3, 2, figsize=(12, 15))

# # Iterate through pairs of resampling techniques
# for i in range(0, len(samplers), 2):
#     sampler_1 = samplers[i]
#     sampler_2 = samplers[i + 1] if i + 1 < len(samplers) else None
    
#     # Apply the first resampling technique and visualize decision boundary
#     X_resampled_1, y_resampled_1 = sampler_1.fit_resample(X_train, y_train)
#     clf_resampled_1 = LogisticRegression()
#     clf_resampled_1.fit(X_resampled_1[['age', 'avg_glucose_level']], y_resampled_1)
#     plot_decision_boundary(X_resampled_1.values, y_resampled_1.values, clf_resampled_1, axs[i // 2, 0], title=f"After {type(sampler_1).__name__}")
    
#     # Apply the second resampling technique and visualize decision boundary if exists
#     if sampler_2:
#         X_resampled_2, y_resampled_2 = sampler_2.fit_resample(X_train, y_train)
#         clf_resampled_2 = LogisticRegression()
#         clf_resampled_2.fit(X_resampled_2[['age', 'avg_glucose_level']], y_resampled_2)
#         plot_decision_boundary(X_resampled_2.values, y_resampled_2.values, clf_resampled_2, axs[i // 2, 1], title=f"After {type(sampler_2).__name__}")

# plt.tight_layout()
# plt.show()


# samplers = [SMOTE(random_state=0), BorderlineSMOTE(random_state=0), KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_clusters=100, n_init=1, random_state=0), random_state=0, cluster_balance_threshold=0.2),
#             SVMSMOTE(random_state=0), ADASYN(random_state=0)]

# # Adjusting subplot to visualize two resampling techniques at a time and before resampling
# fig, axs = plt.subplots(3, 2, figsize=(12, 15))

# # Plot decision boundary before resampling
# clf = LogisticRegression()
# clf.fit(X_train[['age', 'avg_glucose_level']], y_train)
# plot_decision_boundary(X_train.values, y_train.values, clf, axs[0, 0], title="Before Resampling")

# # Iterate through pairs of resampling techniques
# for i in range(0, len(samplers), 2):
#     sampler_1 = samplers[i]
#     sampler_2 = samplers[i + 1] if i + 1 < len(samplers) else None
    
#     # Apply the first resampling technique and visualize decision boundary
#     X_resampled_1, y_resampled_1 = sampler_1.fit_resample(X_train, y_train)
#     clf_resampled_1 = LogisticRegression()
#     clf_resampled_1.fit(X_resampled_1[['age', 'avg_glucose_level']], y_resampled_1)
#     plot_decision_boundary(X_resampled_1.values, y_resampled_1.values, clf_resampled_1, axs[i // 2 + 1, 0], title=f"After {type(sampler_1).__name__}")
    
#     # Apply the second resampling technique and visualize decision boundary if exists
#     if sampler_2:
#         X_resampled_2, y_resampled_2 = sampler_2.fit_resample(X_train, y_train)
#         clf_resampled_2 = LogisticRegression()
#         clf_resampled_2.fit(X_resampled_2[['age', 'avg_glucose_level']], y_resampled_2)
#         plot_decision_boundary(X_resampled_2.values, y_resampled_2.values, clf_resampled_2, axs[i // 2 + 1, 1], title=f"After {type(sampler_2).__name__}")

# plt.tight_layout()
# plt.show()


samplers = [SMOTE(random_state=0), BorderlineSMOTE(random_state=0), KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_clusters=100, n_init=1, random_state=0), random_state=0, cluster_balance_threshold=0.2),
            SVMSMOTE(random_state=0), ADASYN(random_state=0)]

# Adjusting subplot to visualize two resampling techniques at a time and before resampling
fig, axs = plt.subplots(4, 2, figsize=(12, 15))

# Plot decision boundary before resampling with a subset of the data
clf = LogisticRegression()
clf.fit(X_train[['age', 'avg_glucose_level']], y_train)
plot_decision_boundary(X_train.values[:10000], y_train.values[:10000], clf, axs[0, 0], title="Before Resampling")

# Iterate through pairs of resampling techniques
for i in range(0, len(samplers), 2):
    sampler_1 = samplers[i]
    sampler_2 = samplers[i + 1] if i + 1 < len(samplers) else None
    
    # Apply the first resampling technique and visualize decision boundary with a subset of the data
    X_resampled_1, y_resampled_1 = sampler_1.fit_resample(X_train, y_train)
    clf_resampled_1 = LogisticRegression()
    clf_resampled_1.fit(X_resampled_1[['age', 'avg_glucose_level']], y_resampled_1)
    plot_decision_boundary(X_resampled_1.values[:10000], y_resampled_1.values[:10000], clf_resampled_1, axs[i // 2 + 1, 0], title=f"After {type(sampler_1).__name__}")
    
    # Apply the second resampling technique and visualize decision boundary if exists with a subset of the data
    if sampler_2:
        X_resampled_2, y_resampled_2 = sampler_2.fit_resample(X_train, y_train)
        clf_resampled_2 = LogisticRegression()
        clf_resampled_2.fit(X_resampled_2[['age', 'avg_glucose_level']], y_resampled_2)
        plot_decision_boundary(X_resampled_2.values[:10000], y_resampled_2.values[:10000], clf_resampled_2, axs[i // 2 + 1, 1], title=f"After {type(sampler_2).__name__}")
        
# if len(samplers) % 2 != 0:
#     axs[-1, -1].axis('off')

plt.tight_layout()
plt.show()

