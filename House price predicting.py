#!/usr/bin/env python
# coding: utf-8

# # Date importing and first Inspection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


# os
folder_path = "./plots"

if not os.path.exists(folder_path) :
    os.makedirs(folder_path)


# In[3]:


df=pd.read_csv("housing.csv")


# In[4]:


df


# ### Features :
# #### longitude : geographic coordinate ( district east-west  postion)
# #### latitude : geographic coordinat (district north-south postion)
# #### total_rooms : sum of all room in district
# #### population : total population in district
# #### median_income : median household income in district
# #### median_house_value :median house value in district
# #### houseing_median_age : median age of houses in district
# #### ocean_proximity : district's proximity to the ocean

# In[5]:


df.info()


# In[6]:


# there is missing value in total_bedrooms
# all the columns floats exept one column( ocean_proximity) is object


# In[7]:


# show the null value
df[df.total_bedrooms.isna()]


# In[8]:


# check if there any duplicate rows
df[df.duplicated()]


# In[9]:


df.describe()


# In[10]:


df.describe(include="O")


# In[11]:


df.ocean_proximity.unique()


# In[12]:


# NEAR BAY  قرب الخليج
# <1H OCEAN تبعد عن المحيط اقل من ساعة
# INLAND داخل البلاد
# NEAR OCEAN قرب المحيط
# ISLAND جزيرة


# In[13]:


df.ocean_proximity.value_counts()


# In[14]:


df.hist(bins=50,figsize=(12,12))
plot_filename = os.path.join(folder_path , "my_plot.png")
plt.savefig(plot_filename)
plt.show()


# ## Data Cleaning and create additional features

# In[15]:


df.info()


# In[16]:


# dron missing value
df.dropna(inplace=True)


# #### Add New Features

# #### Room per Household

# In[17]:


df["rooms_per_household"]=df.total_rooms.div(df.households)


# In[18]:


df


# In[19]:


df.rooms_per_household.nlargest(10)


# In[20]:


df.rooms_per_household.nsmallest(10)


# In[21]:


df.loc[[5916,8219,1914,1979]]


# #### Population per Household

# In[22]:


df["pop_per_household"]=df.population.div(df.households)


# #### Bedroom per Rooms

# In[23]:


df["bedrooms_per_room"]=df.total_bedrooms.div(df.total_rooms)


# In[24]:


df


# In[25]:


df.describe()


# ### Explanatory Data Analysis

# #### which factors influence house prices?

# In[26]:


df


# In[28]:


df.median_house_value.hist(bins=100,figsize=(12,8))
path_filename=os.path.join(folder_path,"hist_median_house_value.png")
plt.savefig(path_filename)
plt.show()


# In[29]:


# show the corrolation between the columns
df.corr()


# In[35]:


# just we need the correlation between the median house value and the all columns
#  sort the correlation from the postive corrlation to negative correlation
df.corr(numeric_only=True).median_house_value.sort_values(ascending=False)


# In[37]:


# we can see there is high correlation between median_income and median_house_value
# this mean higher income it will be  higher house value
df.median_income.hist(bins=100,figsize=(12,6))
plt.show()


# In[38]:


# regression plot betwwen median_income and median_house_value
# scatter plot with linear Regresssion and histgram for each one
sns.set(font_scale=1.5)
sns.jointplot(data=df,x="median_income",y="median_house_value",kind="reg",height=10)
plt.show()


# In[40]:


# plot with kernel density estimator  مقدر كثافة النواة
sns.set(font_scale=1.5)
sns.jointplot(data=df,x="median_income",y="median_house_value",kind="kde",height=10)
plt.show()


# In[45]:


df.plot(kind="scatter",x="longitude",y="latitude",
       s=df.population/100,label="Population",figsize=(15,10),
       c="median_house_value",cmap="coolwarm",
       colorbar=True,alpha=0.4,fontsize=15,sharex=False)
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("Longitude",fontsize=14)
plt.legend(fontsize=16)
plt.show()


# In[47]:


import matplotlib.image as mpimg
california_img = mpimg.imread("california.png")


# In[48]:


california_img


# In[49]:


plt.figure(figsize=(15,10))
plt.imshow(california_img)
plt.show()


# In[50]:


plt.figure(figsize=(15,10))
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05])
plt.show()


# In[60]:


df.plot(kind="scatter",x="longitude",y="latitude",
       s=df.population/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
path_filename=os.path.join(folder_path,"california_plot.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# In[62]:


prox= df.ocean_proximity.unique()
prox


# ### Plot the data depend on ocean proximity

# In[67]:


df_loc_near_bay= df[df.ocean_proximity == prox[0]].copy()
df_loc_less_one_hour= df[df.ocean_proximity == prox[1]].copy()
df_loc_inland= df[df.ocean_proximity == prox[2]].copy()
df_loc_near_ocean= df[df.ocean_proximity == prox[3]].copy()
df_loc_island= df[df.ocean_proximity == prox[4]].copy()


# #### Naer Bay

# In[79]:


df_loc_near_bay.plot(kind="scatter",x="longitude",y="latitude",
       s=df_loc_near_bay["population"]/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
plt.title("Meadian House Value by Near Bay")
path_filename=os.path.join(folder_path,"california_plot_near_bay.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# #### <1Houre

# In[80]:


df_loc_less_one_hour.plot(kind="scatter",x="longitude",y="latitude",
       s=df_loc_less_one_hour["population"]/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
plt.title("Meadian House Value by <1 Houre")
path_filename=os.path.join(folder_path,"california_plot_less_one_houre.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# ### Inland

# In[81]:


df_loc_inland.plot(kind="scatter",x="longitude",y="latitude",
       s=df_loc_inland["population"]/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
plt.title("Meadian House Value by Inland")
path_filename=os.path.join(folder_path,"california_plot_inland.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# #### Near Ocean

# In[82]:


df_loc_near_ocean.plot(kind="scatter",x="longitude",y="latitude",
       s=df_loc_near_ocean["population"]/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
plt.title("Meadian House Value by Near Ocean")
path_filename=os.path.join(folder_path,"california_plot_near_ocean.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# #### Island

# In[83]:


df_loc_island.plot(kind="scatter",x="longitude",y="latitude",
       s=df_loc_island["population"]/100,c="median_house_value",cmap="coolwarm",
        alpha=0.4,sharex=False,colorbar=True,figsize=(15,10),label="Population"
       )
plt.imshow(california_img,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
plt.legend(fontsize=15)
plt.title("Meadian House Value by Island")
path_filename=os.path.join(folder_path,"california_plot_island.png")
plt.savefig(path_filename, dpi=300,bbox_inches="tight")
plt.show()


# In[84]:


df


# In[86]:


df.median_income.hist(bins=50,figsize=(12,6))
plt.title("Median Income")
plt.show()


# In[87]:


# transform a numeric column (df.median_income) into categorical values based on quantiles (percentiles).
pd.qcut(df.median_income,q=[0,0.25,0.50,0.75,1])


# In[90]:


df["income_cat"]=pd.qcut(df.median_income,q=[0,0.25,0.50,0.75,0.95,1],
                        labels=["Low","Below_Average","Above_Average","High","Very_High"])


# In[92]:


df.income_cat


# In[93]:


df.income_cat.value_counts(normalize=True)


# In[106]:


# plot the categories
plt.figure(figsize=(12,6))
sns.set(font_scale=1.5)
sns.countplot(data=df,x="income_cat",hue="ocean_proximity")
plt.legend(loc=1)
plt.show()


# In[112]:


plt.figure(figsize=(12,6))
sns.set(font_scale=1.5)
sns.barplot(data=df,x="income_cat",y="median_house_value",dodge=True)
plt.show()


# In[113]:


plt.figure(figsize=(12,6))
sns.set(font_scale=1.5)
sns.barplot(data=df,x="ocean_proximity",y="median_house_value",dodge=True)
plt.show()


# In[116]:


df.groupby(["income_cat","ocean_proximity"]).median_house_value.mean().unstack().drop(columns=["ISLAND"])


# In[117]:


matrix=df.groupby(["income_cat","ocean_proximity"]).median_house_value.mean().unstack().drop(columns=["ISLAND"])


# In[118]:


matrix.astype("int")


# In[119]:


plt.figure(figsize=(12,6))
sns.set(font_scale=1.5)
sns.heatmap(matrix.astype(int),cmap="Reds",annot=True,fmt="d",vmin=90000,vmax=470000)
plt.show()


# # Feature Engineering 

# In[120]:


label=df.median_house_value.copy()
label


# In[121]:


features=df.drop(columns=["median_house_value"])
features


# In[122]:


features.info()


# In[123]:


features.select_dtypes("float")


# In[124]:


import scipy.stats as stats


# In[125]:


feat1=features.select_dtypes("float").apply(lambda x:stats.zscore(x))
feat1


# In[126]:


pd.options.display.float_format ="{:.2f}".format


# In[127]:


feat1


# In[128]:


feat1.agg(["mean","std"])


# In[130]:


# handeling the categorecal column
features.ocean_proximity


# In[131]:


features.ocean_proximity.value_counts()


# In[132]:


dummies= pd.get_dummies(features.ocean_proximity)


# In[133]:


dummies


# In[134]:


features = pd.concat([feat1,dummies,df.income_cat],axis=1)


# In[135]:


features


# # Splitting the Data into Train and Test Set

# In[136]:


test_size=0.2


# In[137]:


x_test=features.sample(frac=test_size,random_state=123)


# In[138]:


x_test


# In[139]:


x_test.income_cat.value_counts(normalize=True)


# In[140]:


features.income_cat.value_counts(normalize=True)


# In[141]:


x_test.index


# In[142]:


x_train =features.loc[~features.index.isin(x_test.index)].copy()


# In[144]:


x_train.income_cat.value_counts(normalize=True)


# In[145]:


x_train= x_train.sample(frac=1,random_state=123)


# In[146]:


x_train.drop(columns=["income_cat"],inplace=True)
x_test.drop(columns=["income_cat"],inplace=True)


# In[147]:


y_train = label.loc[x_train.index]
y_test =label.loc[x_test.index]


# # Training the ML Model (Random Forest Regressor)

# In[148]:


from sklearn.ensemble import RandomForestRegressor


# In[149]:


forest_reg=RandomForestRegressor(random_state=42,n_estimators=500,
                                max_features="sqrt",max_depth=75,min_samples_split=2)


# In[150]:


forest_reg.fit(x_train,y_train)


# In[151]:


forest_reg.score(x_train,y_train)


# In[152]:


from sklearn.metrics import mean_squared_error


# In[153]:


pred = forest_reg.predict(x_train)
pred


# In[154]:


forest_mse = mean_squared_error(y_train,pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# # Evaluating the Model on the Test Set

# In[155]:


forest_reg


# In[156]:


forest_reg.score(x_test,y_test)


# In[157]:


pred =forest_reg.predict(x_test)
pred


# In[158]:


forest_mse = mean_squared_error(y_test,pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[159]:


comp = pd.DataFrame(data={"True_v":y_test,"Pred":pred})


# In[160]:


comp


# In[162]:


ae=comp.True_v.sub(comp.Pred).abs()
ae


# In[163]:


mae = ae.mean()


# In[164]:


mae


# # Feature Importance

# In[165]:


forest_reg.feature_importances_


# In[166]:


feature_imp = pd.Series(data=forest_reg.feature_importances_,
                       index=x_train.columns).sort_values(ascending=False)


# In[167]:


feature_imp


# In[169]:


feature_imp.sort_values().plot.barh(figsize=(12,6))
plt.show()


# In[ ]:




