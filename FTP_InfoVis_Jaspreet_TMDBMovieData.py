import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable

from sklearn.preprocessing import StandardScaler

np.random.seed(5764)

#
# # -------------Phase I: Static graphs & Tables-------------
#

loc = 'data/TMDB_movie_dataset_v11.csv'

data = pd.read_csv(loc)


# %%

# Data Cleaning

categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert release_date to datetime format
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Drop rows with NaN release_date
data_cleaned = data.dropna(subset=['release_date'])

# Sort the data by release_date
data_cleaned = data_cleaned.sort_values(by='release_date')

data = data.drop(data[(data['adult'] == True)].index)

data_cleaned['original_language'] = data_cleaned['original_language'].fillna('Unknown')

# Add release year
data_cleaned['release_year'] = data_cleaned['release_date'].dt.year

# Convert numerical features to numeric, coercing errors
numerical_features = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
for feature in numerical_features:
    data_cleaned[feature] = pd.to_numeric(data_cleaned[feature], errors='coerce')

data_cleaned = data_cleaned.dropna(subset=numerical_features)

# Print data types of features
print(data.dtypes)

# Print the features
print("\nCategorical Features:")
print(categorical_features)
print("\nNumerical Features:")
print(numerical_features)

# Check for duplicate data
print("Duplicate data: ", data.drop_duplicates(inplace=True))

# Print missing values
print("\nMissing values in Numerical Data after cleaning:")
print(data_cleaned[numerical_features].isnull().sum())


# %%

# Outlier removal

data_revenueno0 = data_cleaned[(data_cleaned['revenue'] > 0)]
data_non_zero = data_cleaned[(data_cleaned['revenue'] > 0) & (data_cleaned['budget'] != 0)]


# Function to detect and remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Remove outliers from the 'revenue' column
data_no_outliers = remove_outliers_iqr(data_revenueno0, 'revenue')

# Print the number of rows before and after outlier removal
print("\nOutlier removal for Revenue -\n")
print(f"Number of rows before outlier removal: {data_revenueno0.shape[0]}")
print(f"Number of rows after outlier removal: {data_no_outliers.shape[0]}")


# %%

# PCA

from sklearn.decomposition import PCA

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_cleaned[numerical_features])

# Apply PCA
pca = PCA()
pca_fit = pca.fit(data_standardized)
data_pca = pca.transform(data_standardized)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data_pca, columns=[f'PC{i + 1}' for i in range(data_pca.shape[1])])

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(f'\nExplained variance ratio: {[f"{ratio:.2f}" for ratio in explained_variance_ratio]}')

# Singular values
singular_values = pca.singular_values_
print(f'Singular values: {[f"{value:.2f}" for value in singular_values]}')

# Condition number
condition_number = np.linalg.cond(data_standardized)
print(f'Condition number: {condition_number:.2f}')

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Determine the number of components required for 95% variance
num_components_95_variance = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f'Number of components required for 95% variance: {num_components_95_variance}')

# Visualize explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, where='mid',
         label='Cumulative explained variance')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# %%

# Normality tests

from scipy.stats import shapiro, kstest, normaltest


# Define the provided Shapiro-Wilk test function
def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test: {title} feature : statistics = {stats:.2f}, p-value = {p:.2f}')
    alpha = 0.01
    if p > alpha:
        print(f'Shapiro test: {title} is Normal')
    else:
        print(f'Shapiro test: {title} is NOT Normal')


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('=' * 50)
    print(f'K-S test: {title} feature: statistics= {stats:.2f} p-value = {p:.2f}')

    alpha = 0.01
    if p > alpha:
        print(f'K-S test:  {title} is Normal')
    else:
        print(f'K-S test : {title} is Not Normal')
    print('=' * 50)


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('=' * 50)
    print(f'da_k_squared test: {title} feature: statistics= {stats:.2f} p-value = {p:.2f}')

    alpha = 0.01
    if p > alpha:
        print(f'da_k_squaredtest:  {title} is Normal')
    else:
        print(f'da_k_squared test : {title} is Not Normal')
    print('=' * 50)


# Apply the Shapiro-Wilk test to each numerical feature
for feature in numerical_features:
    shapiro_test(data_cleaned[feature], feature)


# %%

# Plot: Line plot for revenue over time

plt.plot(data_cleaned['release_date'], data_cleaned['revenue'], label='Revenue', linewidth=2)
plt.title('Line Plot - Revenue Over Time', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Release Date', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.xlim(pd.Timestamp('1900-01-01'), pd.Timestamp('2024-12-31'))
plt.grid(True)
plt.legend()
plt.show()


# %%

# Bar Plot: Count of Movies per Genre

# Preprocess genres column
data_cleaned['genres'] = data_cleaned['genres'].fillna('')
genres_split = data_cleaned['genres'].str.get_dummies(sep=', ')

# Add genres_split to the original dataframe
data_genres = data_cleaned.join(genres_split)

# Create a dataframe with counts for each genre
genre_counts = genres_split.sum().sort_values(ascending=False).head(10)

# Plot: Stacked bar plot for genres
plt.figure(figsize=(22, 16))
genre_counts.plot(kind='bar', stacked=True, color=sns.color_palette("Paired"))
plt.title('Bar Plot - Count of Movies per Genre', fontsize=30, fontfamily='serif', color='blue')
plt.xlabel('Genres', fontsize=24, fontfamily='serif', color='darkred')
plt.ylabel('Count', fontsize=24, fontfamily='serif', color='darkred')
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()


# %%

# Grouped bar plot

# Explode the genres into separate rows
data_exploded = data_cleaned.assign(genres=data_cleaned['genres'].str.split(', ')).explode('genres')

# Limit to top 10 languages for the plot
top_five_languages = data_exploded['original_language'].value_counts().sort_values(ascending=False).head(5).index
data_top_languages = data_exploded[data_exploded['original_language'].isin(top_five_languages)]

# Limit to top 8 genres for the plot
top_genres = genre_counts.head(8).index
data_top_genres_top_lang = data_top_languages[data_top_languages['genres'].isin(top_genres)]

# Grouped Bar Plot: Count of Movies per Genre per Language
plt.figure(figsize=(14, 10))
sns.countplot(data=data_top_genres_top_lang, x='genres', hue='original_language')
plt.yscale('log')

plt.title('Count of Movies per Genre per Language (Top 5 Languages)', fontsize=18, fontfamily='serif', color='blue')
plt.xlabel('Genres', fontsize=14, fontfamily='serif', color='darkred')
plt.ylabel('Count (Log Scale)', fontsize=14, fontfamily='serif', color='darkred')
plt.legend(title='Original Language')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# %%

# Stacked bar plot

# Filter for the last 10 years (2014 - 2023)
last_10_years = data_exploded[(data_exploded['release_year'] >= 2014) & (data_exploded['release_year'] <= 2023)]

# Limit to top 10 genres for the plot
top_genres_last_ten_years = last_10_years['genres'].value_counts().head(10).index
data_top_genres = last_10_years[last_10_years['genres'].isin(top_genres_last_ten_years)]

# Create a pivot table for the stacked bar plot
pivot_table = data_top_genres.pivot_table(index='release_year', columns='genres', aggfunc='size', fill_value=0)

# Plot: Stacked Bar Plot for Proportion of Genres per Release Year
pivot_table.plot(kind='bar', stacked=True, figsize=(16, 8), colormap='viridis')
plt.title('Proportion of Genres per Release Year (2014 - 2023)', fontsize=18, fontfamily='serif',
          color='blue')
plt.xlabel('Release Year', fontsize=14, fontfamily='serif', color='darkred')
plt.ylabel('Count', fontsize=14, fontfamily='serif', color='darkred')
plt.legend(title='Genres')
plt.grid(True)
plt.show()


# %%

# Count Plot: Movies per Language

top_ten_languages = data_exploded['original_language'].value_counts().head(10).index

plt.figure(figsize=(12, 6))
sns.countplot(data=data_cleaned, x='original_language', order=top_ten_languages)
plt.title('Count of Movies per Original Language (Top 10)', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Original Language', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Count (Log scale)', fontsize=12, fontfamily='serif', color='darkred')
plt.yscale('log')
plt.grid(True)
plt.show()


# %%

# Pie Chart: Distribution of Movies by Genre

plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=120,
        colors=sns.color_palette("Paired"), textprops={'fontsize': 8})
plt.title('Distribution of Movies by Genre', fontsize=14, fontfamily='serif', color='blue')
plt.show()


# %%

# Distribution Plot for Revenue

# Define revenue categories
bins = [0, 2.5e6, 5e6, 7.5e6, 1e7, 2e7, 5e7, 1e8]
labels = ['Under $2.5m', '$2.5m - $5m', '$5m - $7.5m', '$7.5m - $10m', '$10m - $20m', '$20m - $50m', '$50m+']

# Categorize the revenue data into bins
data_revenueno0['revenue_category'] = pd.cut(data_revenueno0['revenue'], bins=bins, labels=labels, right=False)

# Calculate the percentage of movies in each category
category_counts = data_revenueno0['revenue_category'].value_counts(normalize=True).sort_index() * 100

# Plot the bar graph
plt.figure(figsize=(16, 11))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Revenue Distribution', fontsize=22, fontweight='bold', fontfamily='serif', color='blue')
plt.xlabel('Revenue Category', fontsize=16, fontfamily='serif', color='darkred')
plt.ylabel('Percentage of Movies', fontsize=16, fontfamily='serif', color='darkred')
plt.xticks(rotation=30, fontsize=14)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()


# %%

# Histogram plot with KDE

# Remove entries with a vote average of 0
data_cleaned = data_cleaned[data_cleaned['vote_average'] > 0]

plt.figure(figsize=(12, 6))
sns.histplot(data_cleaned['vote_average'], kde=True, color='purple')
plt.title('Distribution of Vote Average', fontsize=18, fontfamily='serif', color='blue')
plt.xlabel('Vote Average', fontsize=14, fontfamily='serif', color='darkred')
plt.ylabel('Frequency', fontsize=14, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# Pair Plot: Numerical Features

pairplot = sns.pairplot(data_non_zero[numerical_features], diag_kind='kde', plot_kws={'alpha': 0.5, 's': 50})
pairplot.fig.suptitle('Pair Plot of Numerical Features', fontsize=20, fontfamily='serif', color='blue')
pairplot.fig.subplots_adjust(top=0.95)  # Adjust top to make room for the suptitle
plt.show()


# %%

# Heatmap with Color Bar

# Create a correlation matrix
corr_matrix = data_non_zero[numerical_features].corr()

# Plot the heatmap with color bar
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontfamily='serif', color='blue')
plt.show()


# %%

# Table for correlation

# Function to determine the relationship strength
def determine_relationship(value):
    if value >= 0.7:
        return "Strong positive relationship"
    elif value >= 0.4:
        return "Moderate positive relationship"
    elif value >= 0.2:
        return "Weak positive relationship"
    elif value <= -0.7:
        return "Strong negative relationship"
    elif value <= -0.4:
        return "Moderate negative relationship"
    elif value <= -0.2:
        return "Weak negative relationship"
    else:
        return "Little to no relationship"


# Create a PrettyTable object
table = PrettyTable()

# Define the column names
table.field_names = ["Feature Pair", "Correlation Coefficient", "Relationship"]

# Add the rows to the table
for i in range(len(numerical_features)):
    for j in range(i + 1, len(numerical_features)):
        feature1 = numerical_features[i]
        feature2 = numerical_features[j]
        corr_value = corr_matrix.loc[feature1, feature2]
        relationship = determine_relationship(corr_value)
        table.add_row([f"{feature1} vs {feature2}", f"{corr_value:.2f}", relationship])

print(table)


# %%

# QQ Plot

from statsmodels.graphics.gofplots import qqplot

data_revenueno0['log_revenue'] = np.log1p(data_revenueno0['revenue'])

# QQ Plot for 'revenue'
plt.figure(figsize=(10, 6))
qqplot(data_revenueno0['log_revenue'], line='45')
plt.title('QQ Plot of Revenue', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Theoretical Quantiles', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Sample Quantiles', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# KDE plot with fill for 'vote_average'

plt.figure(figsize=(10, 6))
sns.kdeplot(data_cleaned['vote_average'], fill=True, alpha=0.6, palette='Blues', linewidth=2)
plt.title('KDE Plot of Vote Average', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Vote Average', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Density', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# Regression plot between Vote Average and Revenue

plt.figure(figsize=(10, 6))
sns.regplot(x='vote_average', y='revenue', data=data_revenueno0, scatter_kws={'s': 10},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Regression Plot of Revenue vs. Vote Average', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Vote Average', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()

# Regression plot between Popularity and Revenue

plt.figure(figsize=(10, 6))
sns.regplot(x='popularity', y='revenue', data=data_revenueno0, scatter_kws={'s': 10},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Regression Plot of Revenue vs. Popularity', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Popularity', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# Create a boxen plot

data_top_languages = data_top_languages[(data_top_languages['revenue'] > 0)]
data_top_languages['log_revenue'] = np.log1p(data_top_languages['revenue'])  # np.log1p is used to handle zero values

plt.figure(figsize=(12, 8))
sns.boxenplot(x='original_language', y='log_revenue', data=data_top_languages)
plt.title('Boxen Plot of Revenue by Original Language', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Original Language', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Log-Transformed Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# %%

# Area plot

data_last_hundred_years = data_revenueno0[
    (data_revenueno0['release_year'] >= 1920) & (data_revenueno0['release_year'] <= 2023)]

revenue_by_year = data_last_hundred_years.groupby('release_year')['revenue'].sum().reset_index()

plt.figure(figsize=(12, 8))
plt.fill_between(revenue_by_year['release_year'], revenue_by_year['revenue'], color="blue", alpha=0.4)
plt.plot(revenue_by_year['release_year'], revenue_by_year['revenue'], color="darkblue", alpha=0.6, linewidth=2)
plt.title('Cumulative Revenue Over Time (Area Plot)', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Total Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# Violin Plot

plt.figure(figsize=(12, 8))
sns.violinplot(x='original_language', y='log_revenue', data=data_top_languages)
plt.title('Violin Plot of Log-Transformed Revenue by Original Language', fontsize=14, fontfamily='serif', color='blue')
plt.xlabel('Original Language', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Log-Transformed Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# %%

# Joint plot with KDE and scatter representation

sns.jointplot(x='vote_average', y='log_revenue', data=data_revenueno0, kind='scatter', s=20, marginal_kws=dict(bins=50))
plt.suptitle('Joint Plot of Vote Average vs. Log-Transformed Revenue', fontsize=14, fontfamily='serif', color='blue')
plt.subplots_adjust(top=0.9)
plt.xlabel('Vote Average', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Log-Transformed Revenue', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()


# %%

# Rug Plot

plt.figure(figsize=(12, 8))

sns.rugplot(x='vote_average', data=data_cleaned, height=0.1, lw=0.5, color='blue')

sns.kdeplot(x='vote_average', data=data_cleaned, fill=True, alpha=0.6, linewidth=2, color='blue')

plt.title('Rug Plot of Vote Average', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vote Average', fontsize=16, fontfamily='serif', color='darkred')
plt.ylabel('Density', fontsize=16, fontfamily='serif', color='darkred')

plt.grid(True)
plt.show()


# %%

# 3d plot and contour plot

from scipy.interpolate import griddata

# Prepare the data for plotting
x = data_revenueno0['vote_average']
y = data_revenueno0['popularity']
z = data_revenueno0['log_revenue']

# Create the figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create grid data
xi = np.linspace(x.min(), x.max(), 1000)
yi = np.linspace(y.min(), y.max(), 1000)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Plot the surface
surf = ax.plot_surface(xi, yi, zi, cmap='coolwarm', edgecolor='none', alpha=0.8)

# Plot the contours
ax.contour(xi, yi, zi, zdir='z', offset=z.min(), cmap='coolwarm', linewidths=1)
ax.contour(xi, yi, zi, zdir='x', offset=x.max(), cmap='coolwarm', linewidths=1)
ax.contour(xi, yi, zi, zdir='y', offset=y.max(), cmap='coolwarm', linewidths=1)

ax.invert_xaxis()
ax.invert_zaxis()

# Customize the axes
ax.set_xlabel('Vote Average', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
ax.set_ylabel('Popularity', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
ax.set_zlabel('Revenue', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})

# Customize the title
ax.set_title('3D Plot of Vote Average, Revenue, and Popularity',
             fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

plt.show()


# %%

# Cluster Map

# Select numerical features for the cluster map
numerical_features_cluster = ['vote_average', 'popularity', 'revenue', 'runtime', 'budget']

# Create a dataframe with selected features
cluster_data = data_revenueno0[numerical_features_cluster]

for feature in numerical_features_cluster:
    cluster_data[feature] = cluster_data[feature].clip(upper=cluster_data[feature].quantile(0.95))

# Standardize the data
scaler = StandardScaler()
cluster_data_standardized = scaler.fit_transform(cluster_data)

# Convert the standardized data back to a DataFrame
cluster_data_standardized = pd.DataFrame(cluster_data_standardized, columns=numerical_features_cluster)

sampled_data = cluster_data_standardized.sample(n=100, random_state=5764)

# Create the cluster map
cluster_map = sns.clustermap(sampled_data, cmap='coolwarm', figsize=(12, 8), linewidths=.5, annot=False)

# Customize the plot
cluster_map.fig.suptitle('Cluster Map of TMDB Movies Numerical Features',
                         fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
cluster_map.fig.subplots_adjust(top=0.95)
plt.show()


# %%

# Hexbin Plot

from matplotlib.colors import LogNorm

# Prepare the data
x = data_revenueno0['vote_average']
y = data_revenueno0['budget']

# Create the hexbin plot
plt.figure(figsize=(10, 8))
hb = plt.hexbin(x, y, gridsize=60, cmap='inferno', mincnt=1, norm=LogNorm())

# Add color bar
cb = plt.colorbar(hb, label='Count (Log scale)')

# Customize the plot
plt.xlabel('Vote Average', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.ylabel('Budget', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.title('Hexbin Plot of Vote Average vs Budget', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
plt.grid(True)
plt.show()


# %%

# Strip Plot

# Create the strip plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='vote_average', y='original_language', data=data_top_genres_top_lang)

# Customize the plot
plt.xlabel('Vote Average', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.ylabel('Original Language', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.title('Strip Plot of Vote Average vs Original Language', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
plt.grid(True)
plt.show()


# %%

# Swarm Plot

plt.figure(figsize=(12, 13))

sampled_data_swarm = data_top_genres_top_lang.sample(n=1500, random_state=5764)

sns.swarmplot(x='genres', y='popularity', data=sampled_data_swarm)

# Customize the plot
plt.xlabel('Genres', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.ylabel('Popularity', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.title('Swarm Plot of Genres vs Popularity', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# %%

# Storytelling Plot

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

data_top_genres_top_lang = data_top_genres_top_lang[
    (data_top_genres_top_lang['release_year'] >= 2014) & (data_top_genres_top_lang['release_year'] <= 2023)]

# Plot 1: Distribution of Genres
genres_count = data_top_genres_top_lang['genres'].value_counts().head(10)
axes[0, 0].pie(genres_count, labels=genres_count.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', 10))
axes[0, 0].set_title('Top Genres Distribution in Movies', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

# Plot 2: Count of Movies per Year
movies_per_year = data_top_genres_top_lang['release_year'].value_counts().sort_index()
axes[0, 1].bar(movies_per_year.index, movies_per_year.values, color=sns.color_palette('viridis', 10))
axes[0, 1].set_title('Count of Movies per Year (Last 10 Years)',
                     fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0, 1].set_xlabel('Year', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0, 1].set_ylabel('Number of Movies', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0, 1].grid(True)

# Plot 3: Distribution of Languages
languages_count = data_top_genres_top_lang['original_language'].value_counts().head(10)
axes[1, 0].bar(languages_count.index, languages_count.values, color=sns.color_palette('viridis', 10))
axes[1, 0].set_title('Top Languages Distributions', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[1, 0].set_xlabel('Language', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1, 0].set_ylabel('Number of Movies', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1, 0].grid(True)

# Plot 4: Average Revenue per Genre
avg_revenue_per_genre = data_top_genres_top_lang.groupby('genres')['revenue'].mean().sort_values(ascending=False).head(
    10)
axes[1, 1].bar(avg_revenue_per_genre.index, avg_revenue_per_genre.values, color=sns.color_palette('viridis', 10))
axes[1, 1].set_title('Top Genres by Average Revenue', fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[1, 1].set_xlabel('Genre', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1, 1].set_ylabel('Average Revenue', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1, 1].grid(True)

# Set the overall title
fig.suptitle('TMDB Movies Analysis (2014-2023) - Storytelling', fontsize=24, fontfamily='serif', color='blue')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
