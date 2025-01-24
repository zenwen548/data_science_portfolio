import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Load datasets
customer_data = pd.read_csv('customers.csv')
sub_data = pd.read_csv('subscriptions.csv')


# -------------------------------
# 1. Data Cleaning
# -------------------------------
def clean_data(df):
    """Cleans the dataset by replacing common null values and removing duplicates."""
    # Replace non-standard null values
    df = df.replace({"NaN": np.nan, "null": np.nan, "": np.nan, "None": np.nan})
    # Remove duplicates
    df = df.drop_duplicates()
    return df


customer_data = clean_data(customer_data)
sub_data = clean_data(sub_data)

# Check for duplicates in subscriptions
print('Duplicate counts:')
print('Subscriptions:', sub_data.duplicated(subset=['customer_id', 'subscription_id']).sum())
print('Customers:', customer_data.duplicated(subset=['customer_id']).sum())

# Deduplicate data
sub_data = sub_data.drop_duplicates(subset=['customer_id', 'subscription_id'])
customer_data = customer_data.drop_duplicates(subset=['customer_id'])


# -------------------------------
# 2. Handle Nulls
# -------------------------------
def handle_nulls(df):
    """Handles null values in the subscription dataset."""
    # Fill numeric fields
    df['subscription_cost'] = df['subscription_cost'].fillna(df['subscription_cost'].mean())
    df['num_support_calls'] = df['num_support_calls'].fillna(0)
    df['referral_code'] = df['referral_code'].fillna(0)
    
    # Convert dates to datetime
    df['subscription_start_date'] = pd.to_datetime(df['subscription_start_date'], errors='coerce')
    df['subscription_end_date'] = pd.to_datetime(df['subscription_end_date'], errors='coerce')
    
    # Calculate tenure days
    today = datetime.today()
    df['tenure_days'] = df.apply(
        lambda row: (row['subscription_end_date'] - row['subscription_start_date']).days
        if pd.notnull(row['subscription_end_date'])
        else (today - row['subscription_start_date']).days,
        axis=1
    )
    
    # Fill average transactions
    df['avg_transactions_per_day'] = df['avg_transactions_per_day'].fillna(df['avg_monthly_transactions'] / 30)
    return df


sub_data = handle_nulls(sub_data)


# -------------------------------
# 3. Standardize Column Names
# -------------------------------
def standardize_columns(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


customer_data = standardize_columns(customer_data)
sub_data = standardize_columns(sub_data)


# -------------------------------
# 4. Merge Datasets
# -------------------------------
merged_data = pd.merge(customer_data, sub_data, on='customer_id', how='right')

# Add derived columns
merged_data['is_active'] = merged_data['subscription_end_date'].isna().astype(int)
merged_data['was_referred'] = merged_data['referral_code'].notnull().astype(int)
merged_data['years_since_signup'] = merged_data.apply(
    lambda row: (
        (row['subscription_end_date'] if pd.notnull(row['subscription_end_date']) else datetime.today()) 
        - row['subscription_start_date']
    ).days / 365.0,
    axis=1
)

# Save cleaned data
merged_data.to_csv('cleaned_data.csv', index=False)


# -------------------------------
# 5. Train Model
# -------------------------------
features = ['tenure_days', 'subscription_cost', 'was_referred', 'num_support_calls', 'avg_monthly_transactions']
target = 'churned'

X = merged_data[features]
y = merged_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -------------------------------
# 6. Predict and Save for Tableau
# -------------------------------
merged_data['predicted_churn'] = model.predict(merged_data[features])
merged_data['predicted_probability'] = model.predict_proba(merged_data[features])[:, 1]
merged_data.to_csv("final_data_for_tableau.csv", index=False)
print("Final dataset saved!")
