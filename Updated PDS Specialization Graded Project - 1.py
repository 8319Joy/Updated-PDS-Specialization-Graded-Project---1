#!/usr/bin/env python
# coding: utf-8

# # 1. Import the required libraries and read the dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


fifa_df = pd.read_csv(r"C:\Users\PURNANGSHU ROY\OneDrive\Desktop\Data\Fifa\fifa.csv")


# In[8]:


fifa_var_df =pd.read_csv(r"C:\Users\PURNANGSHU ROY\OneDrive\Desktop\Data\Fifa variable\fifa_ variable_.csv")


# # 2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features.

# In[9]:


# First few samples
fifa_df.head(5)


# In[10]:


# First few variables
fifa_var_df.head(5)


# In[11]:


#Shape of player info file(rows, columns)
fifa_df.shape


# In[12]:


#Shape of variable file(rows, columns)
fifa_var_df.shape


# In[13]:


# Basic information of the player data file
fifa_df.info()


# In[14]:


# Basic information of the variable file
fifa_var_df.info()


#  Make modification in question two and check for question number 18 or 19 output

# # 3. Drop the columns which you think redundant for the analysis.

# In[16]:


# All Columns in the File
print(fifa_df.columns)


# In[17]:


fifa_df[['Photo','Flag','Club Logo','Jersey Number','Loaned From']].head(5)


# In[18]:


fifa_df.drop(['Photo','Flag','Club Logo', 'Jersey Number', 'Loaned From'],axis=1,inplace=True)
fifa_df.head(5)


# Photo, flag, club logo columns contains image which are irrelevant for analysis. The jersey number column is dropped because it is also irrelevant for player selection and analysis.

# In[19]:


# Shape of  new set after removing redundant columns
fifa_df.shape


# We now have data of 55 features of 18207 players

# # 4. Convert the columns "Value", "Wage", "Release Clause" to float datatype after getting rid of currency symbol and suffix

# In[21]:


# Displaying the columns to be converted
fifa_df[['Value','Wage','Release Clause']].head(5)


# In[22]:


# Removing currency symbol Euro from 'Value', 'Wage' and 'Release Clause' column
fifa_df['Value'] = fifa_df['Value'].str.replace('€','')
fifa_df['Wage'] = fifa_df['Wage'].str.replace('€','')
fifa_df['Release Clause'] = fifa_df['Release Clause'].str.replace('€','')


# In[23]:


# Removing M and K (million and thousand) suffix from currency columns and converting to float datatype
def convert_value(value):
    numeric_part = value.replace(r'[KM]+$', '', regex=True).astype(float)
    multiplier = value.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'],[10**3,10**6]).astype(float)
    return numeric_part * multiplier


# In[24]:


# Modifying 'Value', 'Wage', and 'Release Clause' columns using the defined function
fifa_df['Value'] = convert_value(fifa_df['Value'])
fifa_df['Wage'] = convert_value(fifa_df['Wage'])
fifa_df['Release Clause'] = convert_value(fifa_df['Release Clause'])


# In[25]:


# Displaying modified 'Value', 'Wage', 'Release Clause' columns
fifa_df[['Value','Wage','Release Clause']].head(5)


# # 5. Convert the column "Joined" into integer data type with keeping only the year.
# 

# In[26]:


# Converting "Joined" column to datetype and filtering only the year
fifa_df['Joined'] = pd.to_datetime(fifa_df['Joined']).dt.year


# In[27]:


# Values in the 'Joined' column after converting it to integer
fifa_df['Joined'].unique()


# Initially, the values in "Joined" columns were in Month DD, YYYY format. Now after modifying as per requirements, it is displaying just the years.

# In[28]:


print("The datatype of 'Joined' column is", fifa_df['Joined'].dtype)


# # 6. Convert the column "Contract Valid Until" to pandas datetime type.

# In[29]:


# Displaying uneven format data from "Contract Valid Until"
fifa_df['Contract Valid Until'].unique()


# In[30]:


# Converting "Contract Valid Until" column to pandas datetime type
fifa_df['Contract Valid Until'] = pd.to_datetime(fifa_df['Contract Valid Until'], errors='coerce')


# In[31]:


print('Datatype of "Contract Valid Until" column-', fifa_df['Contract Valid Until'].dtype)


# # 7. The column 'Height' is in inches with a quotation mark, Convert to float with decimal points.

# In[32]:


# 'Height' column and dtype before conversion
fifa_df['Height'].head(5)


# The "Height" column is in inches with "Object" datatype

# In[33]:


# To convert the height data into float format-
# 1) We split the height column into 2 columns with numbers before and after (')
# 2) Joining them together
# 3) Converting the string data into numeric data
fifa_df['Height'] = fifa_df['Height'].str.split("'", expand=True)[0] + '.' + fifa_df['Height'].str.split("'", expand=True)[1]
fifa_df['Height'] = pd.to_numeric(fifa_df['Height'])


# In[34]:


# For better readability of data, rename and add metric unit to the 'Height' column
fifa_df.rename(columns={'Height': 'Height (inch)'}, inplace=True)

# 'Height' column after conversion
fifa_df[['Name','Height (inch)']].head(5)


# In[35]:


print("The datatype of modified 'Height (inch) column is: ", fifa_df['Height (inch)'].dtype)


# # 8. The column "Weight" has the suffix as lbs, remove the suffix and convert to float.

# In[36]:


# 'Weight' column and dtype before conversion
fifa_df['Weight'].head(5)


# The "Weight" column is in lbs with "Object" datatype

# In[38]:


# Change lbs to '' and modify value to float dtype
fifa_df['Weight']=fifa_df['Weight'].str.replace('lbs','').astype(float)

# For better readability,rename and add metric unit in column heading
fifa_df.rename(columns={'Weight': 'Weight (lbs)'}, inplace=True)

# Weight column after conversion
fifa_df[['Name', 'Weight (lbs)']].head(5)


# In[39]:


print("The datatype of modified 'Weight' column is:", fifa_df['Weight (lbs)'].dtype)


# # 9. Check for the percentage of missing values and impute them with appropriate imputation techniques.

# In[40]:


# Checking percentage of missing values for all columns
fifa_df.isnull().sum()/len(fifa_df) * 100


# From the above data we can see that columns like Contract Valid Until, Release Clause have higher percentage of missing values. While most of the columns have low percentage missing values.(0-2%)
# 
# We will follow following steps for imputation

# In[41]:


# Dropping rows of missing values with 0-1%
# Percentage missing values
missing_values = fifa_df.isnull().sum() / len(fifa_df)

# Threshold
threshold = 0.01

# Seperate columns having less than 1% missing value
low_missing_columns = missing_values[missing_values < threshold].index

# Drop rows with low missing values
fifa_df.dropna(subset=low_missing_columns, inplace=True)


# In[42]:


# Seperating datetime columns since mean, median and mode can't be used here
datetime_columns = fifa_df.select_dtypes(include=['datetime64']).columns

# Dropping rows with missing values in datetime columns since the percentage is low
fifa_df.dropna(subset=datetime_columns, inplace=True)


# In[43]:


# Using mean, median and mode on rest of the columns
# Numerical and categorical columns seperation
numerical_columns = missing_values[missing_values > 0].index[fifa_df.dtypes[missing_values > 0] != 'object']
categorical_columns = missing_values[missing_values > 0].index[fifa_df.dtypes[missing_values > 0] == 'object']

# Substituting missing values with median in numerical columns
fifa_df[numerical_columns] = fifa_df[numerical_columns].fillna(fifa_df[numerical_columns].median())

# Substituting missing values with mode in categorical columns
fifa_df[categorical_columns] = fifa_df[categorical_columns].fillna(fifa_df[categorical_columns].mode().iloc[0])


# In[44]:


# Missing values in file after imputation
fifa_df.isnull().sum()/len(fifa_df) * 100


# All the missing values are imputed and the percentage is 0%

# # 10. Plot the distribution of Overall rating for all the players and write your findings.

# In[45]:


# Distribution of Overall rating for all the players using displot
plt.figure(figsize=(10, 6))
sns.distplot(fifa_df['Overall'], bins=20, color='blue', kde=True)
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Density')
plt.grid(False)
plt.show()


# FINDINGS-
# 
# The overall ratings of players range from approximately 46 to 94.
# 
# The ratings are clustered primarily between 60 and 70, indicating that a significant number of players fall within this range.
# 
# The range of ratings among players in the dataset is quite broad, with the lowest overall rating being 46, and the highest being 94.
# 
# The most common rating among the players in the dataset appears to be around 66, as evidenced by the peak.
# 
# On the higher end of the rating scale, there might be some data points that are outliers.
# 
# The distribution of the data shows a long tail that goes beyond the bulk of the dataset, indicating the presence of some data points with extremely high ratings compared to the majority of the data.
# 
# This plot identifies the central tendency of player ratings, the range of ratings, and the potential presence of outliers.

# # 11. Retrieve the names of top20 players based on the Overall rating.

# In[46]:


# Sorting in descending order Name and their overall rating
Top_20_names = fifa_df[['Name','Overall']].sort_values(by = 'Overall', ascending = False).head(20).reset_index()

# Top 20 players with their overall rating
Top_20_names


# # 12. Generate a dataframe which should include all the information of the Top 20 players based on the Overall rating.
# 

# In[47]:


# Filter player data file by matching 'Name' and 'Overall' columns with those in Top_20_names DataFrame.
fifa_df_top20 = fifa_df[(fifa_df['Name'].isin(Top_20_names['Name'])) & 
                        (fifa_df['Overall'].isin(Top_20_names['Overall']))].reset_index(drop=True)

# Required DataFrame
fifa_df_top20


# # 13. What is the average "Age" and "Wage" of these top 20 players?

# In[48]:


# Average age of the top 20 players
average_age = fifa_df_top20['Age'].mean()

# Average wage of the top 20 players
average_wage = fifa_df_top20['Wage'].mean()

print("The average 'age' of top 20 players is", average_age, "year old")
print("The average 'wage' of top 20 players is €" + str(average_wage))


# # 14. Among the top 20 players based on the Overall rating, which player has the highest wage? Display the name of the player with his wage.

# In[49]:


# Filter the top 20 players in fifa_df_top20 from Q.11 by the maximum wage and select only 'Name' and 'Wage' columns to display
print("Player name with his wage:")
fifa_df_top20[fifa_df_top20['Wage']==fifa_df_top20['Wage'].max()][['Name','Wage']]


# # 15. Generate a dataframe which should include the "Player name", "Club Name", "Wage", and 'Overall rating'.

# In[50]:


# Generating DataFrame according to requirement
selected_columns = ['Name', 'Club', 'Wage', 'Overall']
players_club_data = fifa_df[selected_columns]
# Few entries from generated DataFrame
players_club_data.head(5)


# # 15.1) Find the average Overall rating for each club.

# In[51]:


# Average Overall rating for each club
average_overall_ratings = players_club_data.groupby('Club')['Overall'].mean().reset_index()
average_overall_ratings[['Club', 'Overall']]


# # 15.2) Display the average overall rating of Top10 Clubs using a plot

# In[52]:


# Sort DataFrame in descending order of 'Overall', find highest rated clubs
# Top 10 clubs with the highest average overall ratings
top10_clubs = average_overall_ratings.sort_values(by='Overall', ascending=False).head(10)

# Top 10 clubs and their average overall ratings
top10_clubs


# In[53]:


# Horizontal bar plot to show the average overall rating of the top 10 clubs
plt.figure(figsize=(10, 6))
plt.barh(top10_clubs['Club'], top10_clubs['Overall'], color='skyblue')
plt.xlabel('Average Overall Rating', fontsize=14)
plt.ylabel('Club', fontsize=14)
plt.title('Average Overall Rating of Top 10 Clubs', fontsize=14)

# On y-axis plotting names of highest-rated clubs
plt.gca().invert_yaxis()


# # 16. What is the relationship between age and individual potential of the player? Visualize the relationship with appropriate plot and Comment on the same.

# In[54]:


corr_AP = fifa_df[['Age', 'Potential']].corr()
corr_AP


# In[55]:


sns.heatmap(corr_AP, annot=True, cmap='coolwarm')


# In[56]:


# Line plot to visualize the relationship between age and potential
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Potential', data=fifa_df, color='Crimson')
plt.title('Relationship between Age and Potential')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Potential', fontsize=14)
plt.grid(True)
plt.show()


# Comment:
# 
# There is inverse proportionality between age and potential and -0.24 is the correlation coefficient.
# (-0.24) a weak negative linear relationship means with the increasing age their potential decreases , it is also important to note that the relationship is not much strong. So there can be many other factors responsible to decrease in potential
# According to the seaborn line plot, it appears that players tend to achieve their maximum potential at around the age of 17 on average.
# Beyond that, their potential gradually decreases and reaches a relatively stable level until approximately the age of 35.
# However, after the age of 35, there seems to be a noticeable decline in the players' potential.
# Although as said ,there can be other factors responsible for decrease in players potential but here age factor suggests younger players are full of potential than the older ones

# # 17. Which features directly contribute to the wages of the players? Support your answer with a plot and a metric.

# In[57]:


# Selecting relevant features
relevant_features = ['Potential', 'Overall', 'Value', 'International Reputation', 'Release Clause']

# Calculate correlation coefficients
corr_matrix = fifa_df[relevant_features + ['Wage']].corr()

# Select correlation of 'Wage' with relevant features
wage_corr = corr_matrix[['Wage']].sort_values(by='Wage', ascending=False)

# Displaying correlation coefficients
print("Correlation coefficients of relevant features with player wages:")
print(wage_corr)


# This indicates a positive correlation between player wages and these features. Players with higher Value, Release Clause, International Reputation, Overall ratings, and Potentials tend to command higher wages. Among these features, Value and Release Clause show the strongest correlation with Wage.

# In[58]:


# Heatmap - correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(wage_corr, annot=True, cmap='coolwarm')
plt.title('Correlation of Features with Player Wages', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Wage', fontsize=14)
plt.show()


# In[59]:


# Pairplot for studying correlation
sns.set_context("notebook", font_scale=1.2)
sns.pairplot(fifa_df[relevant_features + ['Wage']])
plt.show()


# Comment-
# 
# The 'Wage' column shows strong positive correlations with other significant attributes like 'Potential', 'Overall', 'Value', 'International Reputation', and 'Release Clause', as illustrated by the heatmap and pairplot.
# 
# The aforementioned correlations imply that an increase in these attributes is likely to result in an increase in player wages.
# 
# The attributes that exhibit the most robust correlation with Wage are Value and Release Clause, indicating that players with higher Value and Release Clauses tend to have higher salaries.

# # 18. Find the position in the pitch where the maximum number of players play and the position where the minimum number of players play? Display it using a plot.

# In[60]:


# Position with maximum number of players
max_position_info = fifa_df['Position'].value_counts().idxmax()
max_position_count = fifa_df['Position'].value_counts().max()
print(f"Position with maximum number of players: {max_position_info}, No. of Players: {max_position_count}")


# In[61]:


# Position with minimum number of players
min_position_info = fifa_df['Position'].value_counts().idxmin()
min_position_count = fifa_df['Position'].value_counts().min()
print(f"Position with minimum number of players: {min_position_info}, No. of Players: {min_position_count}")


# In[62]:


min_position = fifa_df['Position'].value_counts().sort_values(ascending=True).head(1)


# In[63]:


# Creating a categorical plot (catplot) to visualize the distribution of player positions
sns.catplot(data=fifa_df, x="Position", kind="count", aspect=2, order=fifa_df['Position'].value_counts().index)

# Adding labels and title
plt.xlabel("Positions")
plt.ylabel("Number of Players")
plt.title("Distribution of Players in Different Positions")
plt.xticks(rotation = 90)
plt.show()


# Inference:
# 
# According to the catplot, the most common position for players in the dataset is 'ST' (Striker), with 2130 players recorded in this role.
# 
# On the other hand, the position with the least amount of players is 'LF' (Left Forward), as only 15 individuals are playing in this position.

# # 19. How many players are from the club 'Juventus' and the wage is greater than 200K? Display all the information of such players.

# In[65]:


# Seperating Juventus players with >200K wage
juventus_high_wage = fifa_df[(fifa_df['Club'] == 'Juventus') & (fifa_df['Wage'] > 200000)].reset_index(drop=True)

# List
juventus_high_wage


# # 20. Generate a data frame containing top 5 players by Overall rating for each unique position.

# In[66]:


# Sort and group DataFrame by 'Position' and 'Overall' and identifying the top 5 players for each position.
top5_players_df = fifa_df.groupby(['Position']).apply(lambda x : x.sort_values('Overall',ascending=False).head(5))

# Resulting few DataFrame samples
top5_players_df.head(15)


# # 21. What is the average wage one can expect to pay for the top 5 players in every position? (use the data frame created in Q19)

# In[67]:


# The average wage for each position
average_wage_by_position = top5_players_df['Wage'].groupby(top5_players_df['Position']).mean()
average_wage_by_position.name = 'Avg Wage'

# Average wage for each position
average_wage_by_position


# In[ ]:




