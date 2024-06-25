#!/usr/bin/env python
# coding: utf-8

# # Cognifiyz  Data Science Internship 

# # Level - 1

# # Task 1: Data Exploration and Preprocessing
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file =  r"C:\Users\Dawood MD\OneDrive\Desktop\Cognifyz\Dataset .csv"


# In[3]:


import chardet
with open (file , 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000000))
result    


# In[4]:


df = pd.read_csv(file,encoding='latin-1',encoding_errors='replace')


# In[5]:


df.head()


# In[7]:


df.tail()


# In[6]:


df.rename(columns={'ï»¿Restaurant ID' : 'Restaurant ID'},inplace=True)


# In[8]:


df.head()


# Explore the dataset and identify the number of rows and columns.

# In[9]:


print('Number fo rows and columns:',df.shape)


# Correction of the corrupted words from the dataset

# In[10]:


# Define a function to replace special characters
def replace_special_characters(text):
    special_char_replacements = {
        'ï¿½': 'i',
        'Û': 'u',
        '±': 'i',
        'Ä': 'A',
        'ç': 'c',
        'Ç': 'C',
        'Ü': 'U',
        'Ö': 'O',
        'Ş': 'S',
        'Ğ': 'G',
        'İ': 'I',
        'ş': 's',
        'ü': 'u',
        'ö': 'o',
        'ğ': 'g',
        'İ': 'I',
        'â': 'a'
    }
    for key, value in special_char_replacements.items():
        text = text.replace(key, value)
    return text


# In[11]:


# Apply the function to relevant columns
columns_to_replace = ['Restaurant Name','City','Address','Locality','Locality Verbose']


# In[12]:


for column in columns_to_replace:
    df[column] = df[column].apply(replace_special_characters)


# In[13]:


df.head(25)


# In[14]:


df.tail()


# Data Information , Description and Datatypes

# In[15]:


df.info()


# In[16]:


df.dtypes


# In[17]:


df.describe()


# Check for missing values in each column and
#  handle them accordingly.

# In[18]:


df.isnull().sum()


# In[19]:


df.dropna(inplace=True)
df.isnull().sum()


# In[20]:


from unidecode import unidecode


# Function to apply unidecode to each text entry
def fix_encoding_issues(text):
    return unidecode(text)

# Apply the function to all string columns in the DataFrame
for column in df.select_dtypes(include=[object]).columns:
    df[column] = df[column].apply(lambda x: fix_encoding_issues(str(x)))

# Display the corrected DataFrame
df.tail()


# In[21]:


# Saving the cleaned data file
df.to_csv('C:/Users/Dawood MD/OneDrive/Desktop/Cognifyz/Cleaned_Dataset1.csv',encoding='UTF-8-SIG')


# In[22]:


df.sample(10)


# Perform data type conversion if necessary.

# In[23]:


# Defining the Dictinary for the Rating and convert it to float values
rating_mapping = {'Excellent' : 5 , 'Very Good' : 4 , 'Good' : 3 , 'Average' : 2 , 'Bad' : 1 , 'Not rated': 0 }

# Applying the custom transformation using lambda function
df['Numeric rating'] = df["Rating text"].apply(lambda x :rating_mapping.get(x))


# In[24]:


df.head()


# Analyze the distribution of the target variable ("Aggregate rating") and identify any class imbalances.

# In[25]:


print(df['Aggregate rating'].value_counts())


# In[26]:


import matplotlib.pyplot as plt

plt.hist(df['Aggregate rating'],bins=[0, 1, 2, 3, 4, 5])
plt.xlabel('Aggregate rating')
plt.ylabel('Number of ratings')
plt.title('Aggregate rating vs. Number of ratings')


# In[27]:


print(df['Rating text'].value_counts())


# From the above outputs we can see that there are 2148 of not rated i,e. Aggregate rating is 0.0 .
# 
# If we wanna to analyze it we have to ignore the 0.0 or not rated data from the datset . 

# In[28]:


df_clean = df[df['Aggregate rating']!= 0.0]


# In[29]:


df_clean


# In[57]:


# Evaluating the cleaned and before cleaned datset 
print("Number of rows and columns in df before cleaning",df.shape)

print("Number of rows and columns in df after cleaning",df_clean.shape)


# In[31]:


print(df_clean["Aggregate rating"].value_counts())


# In[32]:


import matplotlib.pyplot as plt

plt.hist(df_clean['Aggregate rating'],bins=10)
plt.xlabel('Aggregate rating')
plt.ylabel('Number of ratings')
plt.title('Aggregate rating vs. Number of ratings for  cleaned data ')


# In[33]:


plt.boxplot(df_clean['Aggregate rating'])
plt.xlabel('Rating data frequency')
plt.ylabel('Aggregate rating')


# Above Boxplot Concludes that data is Normally Distributed.

# # Task 2:  Descriptive Analysis
#  
#  Calculate basic statistical measures (mean,
#  median, standard deviation, etc.) for numerical
#  columns.

# In[34]:


df[["Average Cost for two","Price range","Aggregate rating","Votes"]].describe()


# Explore the distribution of categorical
#  variables like "Country Code," "City," and
#  "Cuisines."
#  
#  Identify the top cuisines and cities with the
#  highest number of restaurants.

# In[35]:


print("Top Countries 10 having number of Restaurants listed ")
top_countries = df['Country Code'].value_counts().head(10)
top_countries


# In[36]:


import seaborn as sns

sns.countplot(x="Country Code",data = df,palette='rocket')
sns.color_palette("rocket")
plt.title('Distribution of Restaurants  in top 10 countries')
plt.xlabel('Country Codes')
plt.ylabel('Number of Restaurants')


# Country code 1 denotes India , So from the above plot the highest number of restaurants are present in India

# Top Cities

# In[37]:


print("Top Cities  having number of Restaurants listed ")
top_cities = df['City'].value_counts()
top_cities


# In[38]:


plt.figure(figsize=(20,10))
sns.countplot(x="City",data = df,order=df['City'].value_counts().head(20).index,palette='viridis')
sns.color_palette("viridis")
plt.title('Distribution of Restaurants in top 20 countries')
plt.xlabel('Cities')
plt.xticks(rotation=45)
plt.ylabel('Number of Restaurants')


# The above plot finalizes that the highest number of resturants are in Delhi.

# Top Cuisines

# In[39]:


print("Top Cusinies in the Dataset ")
top_cuisines = df['Cuisines'].value_counts()
top_cuisines


# In[40]:


plt.figure(figsize=(20,10))
sns.countplot(x="Cuisines",data = df,order=df['Cuisines'].value_counts().head(30).index,palette="cubehelix")
sns.color_palette("cubehelix")
plt.title('Distribution of Top 30 Cuisines')
plt.xlabel('Cuisines')
plt.xticks(rotation=90)
plt.ylabel('Counts')


# # Task 3

# Geospatial Analysis Visualize the locations of restaurants on a map using latitude and longitude information.
# 

# In[41]:


get_ipython().system('pip install geopandas')
get_ipython().system('pip install shapely')


# In[42]:


#Importing the necessary Libraries
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import point


# In[43]:


geo_df = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.Longitude,df.Latitude))
world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
geo_df.plot(ax=world_map.plot("continent" ,legend =True,figsize =(30,15)),marker ="*" ,color = 'blue')
plt.show()


# Analyze the distribution of restaurants across different cities or countries.
# Determine if there is any correlation between the restaurant's location and its rating

# In[44]:


plt.figure(figsize=(20,10))
sns.countplot(y="City",data = df,order=df['City'].value_counts().head(20).index,palette='viridis')
sns.color_palette("viridis")
plt.title('Distribution of Restaurants in top 20 countries')
plt.xlabel('Cities')
plt.ylabel('Number of Restaurants')


# In[45]:


plt.figure(figsize=(8,4))
corr = df[["Longitude","Latitude","Aggregate rating"]].corr()
sns.heatmap(corr, cmap="coolwarm",fmt='.2f',annot=True)
plt.title('Correlation between Restuarants location and Aggregate rating')
plt.show()


# # Level 2

# Task 1 :
# Table Booking and Online Delivery Determine the percentage of restaurants that offer table booking and online delivery.
# 

# In[46]:


# Percentage of restaurants offers table bookings are as follows
table_booking_percentage = (df["Has Table booking"].value_counts()/len(df))*100
print(table_booking_percentage)
print("The percentage of Restaurants that has the table booking:",table_booking_percentage['Yes'],"%")


# In[47]:


table_booking_labels = ['Does not offer table Booking','Offers Table booking']
plt.pie(table_booking_percentage,explode=(0.1,0.1),labels=table_booking_labels,autopct='%1.1f%%',colors=['#ff9999','#66b3ff'],shadow=True)


# In[48]:


# Percentage of restaurants offers Online delivery are as follows
online_delivery_percentage = (df["Has Online delivery"].value_counts()/len(df))*100
print(online_delivery_percentage)
print("The percentage of Restaurants that has the Online delivery:",online_delivery_percentage['Yes'],"%")


# In[49]:


online_delivery_labels = ['Does not has Online Delivery','Has Online Delivery']
plt.pie(online_delivery_percentage,explode=(0.1,0.1),labels=online_delivery_labels,autopct='%1.1f%%',colors=['gray', 'saddlebrown'],shadow=True,)


# Compare the average ratings of restaurants with table booking and those without.

# In[50]:


#Calculating the average rating of restaurants with table booking and those without
avg_rating_with_table_booking = df[df["Has Table booking"]=='Yes']["Aggregate rating"].mean()
avg_rating_without_table_booking = df[df["Has Table booking"]=='No']["Aggregate rating"].mean()

#Plotting Essentials
categories = ['With Table Booking','Without Table booking']
avg_ratings = [avg_rating_with_table_booking,avg_rating_without_table_booking]

#Plotting the Bar Graph
plt.bar(categories,avg_ratings,color =['#66b3ff','#ff9999',] )
plt.xlabel('Table Booking')
plt.ylabel("Average Rating")
plt.title('Plot Between Table booking Vs. Average Rating')
plt.show()


# Analyze the availability of online delivery among restaurants with different price ranges.

# In[51]:


#Categories the Data into price range and calculate the proportion of online delivery
price_ranges = df['Price range'].unique()
availability = []

for range in price_ranges:
    total_range = df[df["Price range"] == range].shape[0]
    online_delivery_range = df[(df['Price range'] == range) & (df['Has Online delivery'] == 'Yes')].shape[0]
    proportion = online_delivery_range / total_range if total_range > 0 else 0
    availability.append(proportion)
    
# Preparing the Data Essentials for Plotting
categories = price_ranges
proportions = availability

#Plotting the Bar Graph
plt.bar(categories,proportions,color =['#66b3ff','#ff9999',"#99ff99"] )
plt.xlabel('Price range')
plt.ylabel("Restaurants Ofeering Online Delivery")
plt.title('Plot Between Online Delivery Availability Vs. Price range')
plt.show()


# Task 2 : Price Range Analysis
# Determine the most common price range among all the restaurants.

# In[52]:


#counting the occurence of each price range
price_range_counts = df["Price range"].value_counts()

# Determining the most common price range among all the restaurants
most_common_price_range = price_range_counts.idxmax()
most_common_count = price_range_counts.max()

#Visualization of the distribution of the price range
price_range_counts.plot(kind ="bar",color ="indigo")
plt.xlabel('Price range')
plt.ylabel("Number of Restuarnts")
plt.title('Distribution of Price range of the Restaurants')
plt.show()

print(f'The Most Common price of the restuarnts is:{most_common_price_range} with {most_common_count} Restaurants')


# Calculate the average rating for each price range.Identify the color that represents the highest average rating among different price ranges.
# 

# In[53]:


#calculate the average ratings for each price range
average_ratings = df.groupby("Price range")["Aggregate rating"].mean().reset_index()

print(average_ratings)

# visualize the average ratings with the colors
plt.bar(average_ratings["Price range"],average_ratings["Aggregate rating"],color =['blue','red','yellow','green'])
plt.xlabel('Price range')
plt.ylabel("Aggregate rating")
plt.title('Average rating Vs. Price range')
plt.xticks(rotation = 45)
plt.show()


# Task 3: Feature Engineering
# Extract additional features from the existing columns, such as the length of the restaurant name or address.

# In[54]:


#Extracting the length of the restaurant names
df['Name length'] = df['Restaurant Name'].apply(len)

#Extracting the legth of the address 
df['Address length'] = df['Address'].apply(len)

df.head(25)


# Create new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables

# In[55]:


# Creationg the new feature for table booking and online delivery
df['has table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['has online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

df.head(25)


# In[56]:


#save the Updated Dataframe into the New csv file
df.to_csv(r"C:\Users\Dawood MD\OneDrive\Desktop\Cognifyz\updated_resturant_Dataset.csv")


# # Thank you

# In[ ]:




