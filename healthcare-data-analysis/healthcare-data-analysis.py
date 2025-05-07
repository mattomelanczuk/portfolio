"""
Heart Disease Prediction using Machine Learning

Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for 31% of all deaths.
This project uses a dataset with 11 features to predict the likelihood of heart disease. Early detection
and management of heart disease can save lives by enabling timely medical intervention.

Key Features:
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning for Predictive Analytics
"""

import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "heart.csv"  # File name from the Kaggle dataset
cache_file = "heart.csv"  # Local cache file name

if os.path.exists(cache_file):  # Check if the cache file exists
    print("Loading data...")
    df = pd.read_csv(cache_file)  # If it exists, load the data from the cache file
else:
    print("Fetching data from Kaggle...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/heart-failure-prediction",
        file_path,
    )
    if not os.path.exists(cache_file):
        df.to_csv(cache_file, index=False)
        print(f"Data cached to {cache_file}")
        
# ======================================== Data Preprocessing ======================================== #
        
print(df, end="\n\n")
print(df.head(), end="\n\n")
print(df.columns, end="\n\n")
print(df.dtypes, end="\n\n")
print("Descriptive Statistics:")
print(df.describe(include='all'))

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

missing_values = df.isnull().sum()
print(f'\nMissing values in each column:\n{missing_values}')

# ========================================= Data Cleaning ======================================== #
# zero_value_cols = ['RestingBP', 'Cholesterol']  # Handle zero values in restingbp and cholesterol

# for col in zero_value_cols:
#     if col in df.columns:
#         zero_count = (df[col] == 0).sum()
#         print(f"Number of zero values in {col}: {zero_count}")
        
#         # Replace zeros with the median of the column
#         if zero_count > 0:
#             median_value = df.loc[df[col] != 0, col].median()
#             df[col] = df[col].replace(0, median_value)
#             print(f"Replaced zero values in {col} with median value: {median_value}")
            
# # Summary of changes
# print("\nSummary of zero-value handling:")
# for col in zero_value_cols:
#     if col in df.columns:
#         print(f"{col}: Zero values replaced with median, and binary indicator added as {col}_was_zero.")

# ======================================== Data Visualization ======================================== #
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Identify continuous and binary categorical columns
continuous_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Binary categorical columns: only 0 or 1 values
binary_cols = [col for col in continuous_cols if set(df[col].dropna().unique()).issubset({0, 1})]
# Continuous columns: exclude binary columns
cont_cols = [col for col in continuous_cols if col not in binary_cols]

# Plot continuous columns
num_cont = len(cont_cols)
fig1, axes1 = plt.subplots(nrows=(num_cont + 3) // 3, ncols=3, figsize=(15, 3 * ((num_cont + 3) // 3)))
axes1 = axes1.flatten()
palette = sns.color_palette("mako", num_cont)

for i, col in enumerate(cont_cols):
    ax = axes1[i]
    hist = sns.histplot(df[col], kde=True, ax=ax, color=palette[i % len(palette)], edgecolor='black')
    ax.set_title(col)
    # Add count labels
    for patch in hist.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (patch.get_x() + patch.get_width() / 2, height), 
                        ha='center', va='bottom', fontsize=7, color='black', rotation=0)

for j in range(i + 1, len(axes1)):
    fig1.delaxes(axes1[j])

plt.tight_layout()
sns.despine()
plt.show()

# Plot binary categorical columns as countplots
num_bin = len(binary_cols)
if num_bin > 0:
    fig2, axes2 = plt.subplots(nrows=(num_bin + 2) // 2, ncols=2, figsize=(12, 2 * ((num_bin + 2) // 2)))
    axes2 = axes2.flatten()
    bin_palette = sns.color_palette("mako", num_bin)
    for i, col in enumerate(binary_cols):
        cp = sns.countplot(x=col, data=df, ax=axes2[i], hue=col, palette=bin_palette, legend=False)
        axes2[i].set_title(col)
        axes2[i].set_xticks([0, 1])
        # Add count labels inside bars and bold
        for p in cp.patches:
            height = p.get_height()
            if height > 0:
                cp.annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height / 2),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold'
                )
    for j in range(i + 1, len(axes2)):
        fig2.delaxes(axes2[j])
    plt.tight_layout()
    sns.despine()
    plt.show()
else:
    print("No binary categorical columns found.")
    
# Plot non-binary categorical columns as countplots
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Also include int columns with few unique values (excluding binary)
cat_cols += [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                if 2 < df[col].nunique() <= 10 and col not in binary_cols]

cat_cols = list(set(cat_cols))  # Remove duplicates

num_cat = len(cat_cols)
if num_cat > 0:
    fig3, axes3 = plt.subplots(nrows=(num_cat + 3) // 3, ncols=3, figsize=(15, 3 * ((num_cat + 3) // 3)))
    axes3 = axes3.flatten()
    cat_palette = sns.color_palette("mako", num_cat)
    for i, col in enumerate(cat_cols):
        unique_vals = df[col].nunique()
        palette_for_col = sns.color_palette("mako", unique_vals)
        cp = sns.countplot(x=col, data=df, ax=axes3[i], hue=col, palette=palette_for_col, legend=False)
        axes3[i].set_title(col)
        # Add count labels inside bars and bold
        for p in cp.patches:
            height = p.get_height()
            if height > 0:
                cp.annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height / 2),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold'
                )
    for j in range(i + 1, len(axes3)):
        fig3.delaxes(axes3[j])
    plt.tight_layout()
    sns.despine()
    plt.show()
else:
    print("No non-binary categorical columns found.")

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", square=True, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# ======================================== Feature Importance with Machine Learning ======================================== #

# Assume the target column is named 'HeartDisease' or similar; adjust as needed
target_col = None
for col in df.columns:
    if col.lower() in ['heartdisease']:
        target_col = col
        break

if target_col is None:
    raise ValueError("Could not automatically detect the target column. Please set 'target_col' manually.")

# Encode categorical features if any
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\nTop contributing factors to heart disease (feature importances):")
for feature, importance in feat_importance.items():
    print(f"{feature}: {importance:.4f}")

# Plot feature importances with data labels inside bars
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feat_importance.values, y=feat_importance.index, hue=feat_importance.index, palette="viridis", legend=False)
plt.title("Feature Importances for Heart Failure Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# Add data labels inside bars
for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.annotate(f'{width:.2f}',
                    (width / 2, p.get_y() + p.get_height() / 2),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')

plt.show()

print("\nTop 5 contributing factors to heart disease (by feature importance):")
for feature, importance in feat_importance.head(5).items():
    print(f"{feature}: {importance:.4f}")
    
# ======================================== Model Evaluation ======================================== #
# Predict on test set
y_pred = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nModel Summary:")
print(f"Model Type: Random Forest Classifier")
print(f"Number of Features: {X.shape[1]}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")
print(f"Test Accuracy: {accuracy:.4f}")
print("Top 5 Features by Importance:")
for feature, importance in feat_importance.head(5).items():
    print(f"  {feature}: {importance:.4f}")
    
# Analyze the contribution of 'ST_Slope' values to heart disease
if 'ST_Slope' in df.columns and target_col is not None:
    st_slope_impact = df.groupby('ST_Slope')[target_col].mean().sort_values(ascending=False)
    print("\nAverage heart disease rate by ST_Slope value (descending):")
    print(st_slope_impact)
    most_risky_st_slope = st_slope_impact.idxmax()
    print(f"\nST_Slope value most associated with heart disease: {most_risky_st_slope} (rate: {st_slope_impact.max():.2f})")
else:
    print("\n'ST_Slope' column not found in the dataset.")
    
# Analyze the contribution of the other top 5 features to heart disease
top_features = feat_importance.head(5).index.tolist()
# Exclude 'ST_Slope' if present, since already analyzed
top_features = [f for f in top_features if f != 'ST_Slope']

for feature in top_features:
    unique_vals = df[feature].nunique()
    if unique_vals <= 10 or df[feature].dtype == 'object':
        rates = df.groupby(feature)[target_col].mean().sort_values(ascending=False)
        most_risky = rates.idxmax()
        print(f"\n{feature} value most associated with heart disease: {most_risky} (rate: {rates.max():.2f})")
    else:
        median_val = df[feature].median()
        above = df[df[feature] > median_val][target_col].mean()
        below = df[df[feature] <= median_val][target_col].mean()
        if above > below:
            print(f"\n{feature} > {median_val:.2f} is most associated with heart disease (rate: {above:.2f})")
        else:
            print(f"\n{feature} <= {median_val:.2f} is most associated with heart disease (rate: {below:.2f})")
