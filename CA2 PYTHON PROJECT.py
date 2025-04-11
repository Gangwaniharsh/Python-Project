import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOADING DATASET
dataset=pd.read_csv("C:/Users/mi/Downloads/Crime_Data_from_2020_to_present (1).csv")
print(dataset)

#EXPLORING DATASET
print("Information: \n",dataset.info())
print("Description: \n",dataset.describe())

#HANDLING MISING VALUES
print("Missing values ",dataset.isnull().sum())

#Remove duplicates
dataset.drop_duplicates(inplace=True)

#Clean column names
dataset.columns = dataset.columns.str.strip()

# Fill object (text) columns with 'PYTHON'
obj_cols = ['Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc', 'Cross Street', 'Mocodes']
for col in obj_cols:
    dataset[col] = dataset[col].fillna('PYTHON')

# Fill numeric columns with median
dataset['Weapon Used Cd'] = dataset['Weapon Used Cd'].fillna(dataset['Weapon Used Cd'].median())
dataset['Crm Cd 1'] = dataset['Crm Cd 1'].fillna(dataset['Crm Cd 1'].median())

#`Crm Cd 2`, `Crm Cd 3`, `Crm Cd 4` are not important,drop them
dataset.drop(columns=['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4'], inplace=True)

# 5. Final check
print("Final DataFrame shape:", dataset.shape)
print("\nRemaining missing values:\n", dataset.isnull().sum())
print("\nDataset Info:")

dataset.info()

# Count the frequency of each crime type
crime_counts = dataset['Crm Cd Desc'].value_counts()

# Get top 10 most frequent crime types
crime_counts = dataset['Crm Cd Desc'].value_counts().head(10).reset_index()
crime_counts.columns = ['Crime Type', 'Count']

# Add a dummy hue column to satisfy seaborn's future requirement
crime_counts['Hue'] = crime_counts['Crime Type']

# Plot using seaborn with hue and palette
plt.figure(figsize=(12, 6))
sns.barplot(
    data=crime_counts,
    x='Count',
    y='Crime Type',
    hue='Hue',
    palette='viridis',
    dodge=False,
    legend=False  # disables legend since hue is just for coloring
)
plt.title('Top 10 Most Common Crime Types (2020–Present)', fontsize=14)
plt.xlabel('Number of Incidents')
plt.ylabel('Crime Type')
plt.tight_layout()
plt.show()

# Convert DATE OCC to datetime
dataset['DATE OCC'] = pd.to_datetime(dataset['DATE OCC'], errors='coerce')

# Drop rows with invalid dates
dataset = dataset.dropna(subset=['DATE OCC'])

# Extract year and month
dataset['Year'] = dataset['DATE OCC'].dt.year
dataset['Month'] = dataset['DATE OCC'].dt.month

# Crimes Per Year 
crimes_per_year = dataset['Year'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.lineplot(
    x=crimes_per_year.index,
    y=crimes_per_year.values,
    marker='o',
    color='steelblue'  # Use a solid color to avoid palette warnings
)
plt.title('Total Crimes Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()

# Crimes Per Month (All Years Combined) 
crimes_per_month = dataset['Month'].value_counts().sort_index()

# Month number to name mapping
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(10, 5))
sns.barplot(
    x=month_labels,
    y=crimes_per_month.values,
    color='skyblue'  # Use a single color to avoid using palette
)
plt.title('Total Crimes Per Month (All Years Combined)')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.tight_layout()
plt.show()

dataset = dataset.dropna(subset=["LAT", "LON"])

dataset['DATE OCC'] = pd.to_datetime(dataset['DATE OCC'], errors='coerce')

# Extract time features
dataset['Hour'] = dataset['TIME OCC'] // 100
dataset['Month'] = dataset['DATE OCC'].dt.month
dataset['Year'] = dataset['DATE OCC'].dt.year
dataset['DayOfWeek'] = dataset['DATE OCC'].dt.day_name()

#Crimes by Hour
plt.figure(figsize=(10, 4))
sns.countplot(data=dataset, x='Hour', hue='Hour', palette='magma', legend=False)
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Crime Count by Day of the Week
plt.figure(figsize=(10, 4))
sns.countplot(data=dataset, x='DayOfWeek', hue='DayOfWeek', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='viridis', legend=False)
plt.title("Crimes by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 Crime Types
top_crimes = dataset['Crm Cd Desc'].value_counts().nlargest(10).reset_index()
top_crimes.columns = ['Crm Cd Desc', 'Count']
plt.figure(figsize=(10, 4))
sns.barplot(data=top_crimes, y='Crm Cd Desc', x='Count', hue='Crm Cd Desc', palette='coolwarm', legend=False)
plt.title("Top 10 Reported Crimes")
plt.xlabel("Count")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

# Crimes by Month
plt.figure(figsize=(10, 4))
sns.countplot(data=dataset, x='Month', hue='Month', palette='Blues', legend=False)
plt.title("Crimes by Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# Select numeric columns
numeric_dataset = dataset.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_dataset.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# Prepare necessary fields
dataset['DATE OCC'] = pd.to_datetime(dataset['DATE OCC'], errors='coerce')
dataset['Hour'] = dataset['TIME OCC'] // 100
dataset['DayOfWeek'] = dataset['DATE OCC'].dt.day_name()

# Filter top 5 most common crimes
top_crimes = dataset['Crm Cd Desc'].value_counts().nlargest(5).index
subset = dataset[dataset['Crm Cd Desc'].isin(top_crimes)]

# Victim Age by Crime Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=subset, x='Crm Cd Desc', y='Vict Age', hue='Crm Cd Desc', palette='Set3', legend=False)
plt.title("Victim Age Distribution by Crime Type")
plt.xlabel("Crime Type")
plt.ylabel("Victim Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Victim Age by Gender
plt.figure(figsize=(8, 5))
sns.boxplot(data=dataset, x='Vict Sex', y='Vict Age', hue='Vict Sex', palette='coolwarm', legend=False)
plt.title("Victim Age by Gender")
plt.xlabel("Victim Gender")
plt.ylabel("Victim Age")
plt.tight_layout()
plt.show()


# Victim Age Distribution KDE
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dataset, x='Vict Age', fill=True, color='skyblue')
plt.title("Victim Age Distribution (KDE)")
plt.xlabel("Victim Age")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# Victim Age KDE by Gender
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dataset, x='Vict Age', hue='Vict Sex', fill=True, common_norm=False, palette='Set2')
plt.title("Victim Age Distribution by Gender")
plt.xlabel("Victim Age")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

