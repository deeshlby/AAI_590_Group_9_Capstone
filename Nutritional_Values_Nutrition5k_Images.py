#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Acknolwledgement: referencing Stackoverflow.com and chatgpt in data interrpretability and code debugging 


# In[179]:


# Importing Libraries
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[131]:


ls


# In[132]:


pwd


# # Data Exploration

# In[180]:


# we set the paths of the imagery files and subset from the metadata folder, dish_metadata_cafe1.csv
imagery_path = '/Users/dina/Downloads/nutrition5k_dataset/imagery/realsense_overhead'
metadata_path = '/Users/dina/Downloads/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv'


# In[181]:


# we choose to use the first six features of the columns
df_labels = pd.read_csv(metadata_path, usecols=[0, 1, 2, 3, 4, 5])


# In[182]:


print(df_labels.head(20))


# In[183]:


print(df_labels.shape)


# In[184]:


# we add the column names of the first six columns, from the readme of the Nutrition5k 
# https://github.com/google-research-datasets/Nutrition5k
# ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
# choosing to focus our exploration towards the identification of the nutritional facts 
# of the corresponding images of dishes with corresponding dish_id in the metadata.
# knowing the nutritionnal facts, and the caloric category of the meal  would be of great help in meal choice.
df_labels.columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']


# In[185]:


print(df_labels.head())


# In[186]:


# we check if there are missing values
print('missing values per column:')
print(df_labels.isna().sum())


# In[187]:


# we print some stats of the metadata
print('summary stats')
print(summary_stats)


# # Visualization

# In[150]:


# We create a for loop to plot histograms for the distribution of each fearute

plt.figure(figsize=(15, 10))

for i, column in enumerate(['total_calories', 'total_mass', 'total_carb', 'total_protein'], 1):
    plt.subplot(2, 3, i)
    plt.hist(df_labels[columns], bins=30, edgecolor='k')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show


# In[149]:


# we use a box plot ot all nutritional values
plt.figure(figsize=(15, 5))
df_labels[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']].boxplot()
plt.title('Box plot nutritional values')
plt.ylabel('values')
plt.show


# In[152]:


# we create a correlation matrix to the different features
correlation_matrix = df_labels[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']].corr()
print(correlation_matrix)


# In[153]:


# we create a correlation heatmap for better visualization of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Nutritional Information')
plt.show


# In[ ]:





# # Feature Engineering

# In[188]:


#  we categorize the 'total_calories' feature( which is continuous data)
# into bins labeled(low, medium and high) inside a new column 'calorie_category'
df_labels['calorie_category'] = pd.cut(
    df_labels['total_calories'],
    bins=[0, 200, 500, float('inf')],
    labels=['Low', 'Medium', 'High'])


# In[189]:


# we plot the distribution of the different calorie categories
sns.countplot(x='calorie_category', data=df_labels)
plt.title('Distribution of Calorie Categories')
plt.xlabel('Calorie Category')
plt.ylabel('Count')
plt.show()


# # Pre-Processing

# In[191]:


# We Standardize all numerical values
scaler = StandardScaler()
df_labels[['total_calories', 'toal_mass', 'total_fat', 'total_carb', 'total_protein']] = scaler.fit_transform(
    df_labels[['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']])


# In[195]:


df_labels


# # Associating the images with labels

# In[221]:


# we  merge the imagery/realsense_overhead folders with the df_labels dataframe 
# we create a for loop to iterate through the imagery folders and match them with the dish_id
image_data= []

for folder_name in os.listdir(imagery_path):
    folder_path = os.path.join(imagery_path, folder_name)
    
    if os.path.isdir(folder_path):
        matching_row = df_labels[df_labels['dish_id'] == folder_name]
        
        if not matching_row.empty:
            labels_info = matching_row.iloc[0].to_dict()
            
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name == 'rgb.png':
                    # Open the image and convert to RGB
                    image = Image.open(file_path).convert('RGB')
                    
                    image_data.append({'image': image, 'label_info': labels_info})


# In[222]:


len(image_data)


# In[223]:


image_data[1]


# In[224]:


# we convert image_data to a dataframe
df_image_data = pd.DataFrame(image_data)
print(df_image_data.head())


# In[225]:


# we will plot random sample of images from df_image_data with their corresponding metadata to prove association
fig, axes = plt.subplots(5, 1, figsize=(10, 25))

create displayed_dishes_ids to store the sample dishes and avoid repeats

displayed_dish_ids = set()


# we create a  loop to display 5 samples with images
for i in ranges(5):
    while True:
  
        sample_row = df_image_data.sample(1).iloc[0] # we select a random row from df_image_data
        sample_dish_id = sample_row['label_info']['dish_id']
       
        
        if sample_dish_id in displayed_dish_ids:
            continue
            
        displayed_dish_ids.add(sample_dish_id)
        
        image = sample_row['image']
        label_info = sample_row['label_info']
        
        axes[i].imshow(image)
        axes[i].axis('off')
        
        title = (
             f"Dish ID: {sample_dish_id}\n"
            f"Calories: {label_info['total_calories']} kcal, "
            f"Mass: {label_info['total_mass']} g\n"
            f"Fat: {label_info['total_fat']} g, "
            f"Carbs: {label_info['total_carb']} g, "
            f"Protein: {label_info['total_protein']} g"
        )
        axes[i].set_title(title, fontsize=12)
        
        break
        
plt.tight_layout()
ply.show()


# In[ ]:




