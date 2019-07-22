#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns 


# In[4]:


df=pd.read_csv("all.csv", header=None,
               names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],
)
df.head()


# In[5]:


df.dtypes


# In[6]:


df.shape


# In[7]:


df.shape[0], df.shape[1]


# In[8]:


df.columns


# In[9]:


type(df.rating), type(df)


# In[10]:


df.rating < 3


# In[11]:


np.sum(df.rating < 3)


# In[13]:


print (1*True, 1*False)


# In[14]:


np.sum(df.rating < 3)/df.shape[0]


# In[15]:


np.sum(df.rating < 3)/float(df.shape[0])


# In[16]:


np.mean(df.rating < 3.0)


# In[17]:


(df.rating < 3).mean()


# In[18]:


df.query("rating > 4.5")


# In[19]:


df[df.year < 0]


# In[20]:


df[(df.year < 0) & (df.rating > 4)]#there were none greater than 4.5!


# In[21]:


df.dtypes


# In[22]:


df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int)


#  Its trying to convert some python datatype: None into an int. This usually means data was missing.

# In[23]:


df[df.year.isnull()]


# In[24]:


df = df[df.year.notnull()]
df.shape


# In[25]:


df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int)


# In[26]:


df.dtypes


# In[27]:


df.rating.hist();


# In[29]:


sns.set_context("notebook")
meanrat=df.rating.mean()
#you can get means and medians in different ways
print (meanrat, np.mean(df.rating), df.rating.median())
with sns.axes_style("whitegrid"):
    df.rating.hist(bins=30, alpha=0.4);
    plt.axvline(meanrat, 0, 0.75, color='r', label='Mean')
    plt.xlabel("average rating of book")
    plt.ylabel("Counts")
    plt.title("Ratings Histogram")
    plt.legend()
    #sns.despine()


# In[30]:


df.review_count.hist(bins=np.arange(0, 40000, 400))


# In[31]:


df.review_count.hist(bins=100)
plt.xscale("log");


# In[32]:


plt.scatter(df.year, df.rating, lw=0, alpha=.08)
plt.xlim([1900,2010])
plt.xlabel("Year")
plt.ylabel("Rating")


# In[33]:


alist=[1,2,3,4,5]


# In[34]:


asquaredlist=[i*i for i in alist]
asquaredlist


# In[35]:


plt.scatter(alist, asquaredlist);


# In[37]:


print (type(alist))


# In[38]:


plt.hist(df.rating_count.values, bins=100, alpha=0.5);


# In[40]:


print (type(df.rating_count), type(df.rating_count.values))


# Vectorization

# In[41]:


alist + alist


# In[42]:


np.array(alist)


# In[43]:


np.array(alist)+np.array(alist)


# In[44]:


np.array(alist)**2


# In[45]:


newlist=[]
for item in alist:
    newlist.append(item+item)
newlist


# In[47]:


a=np.array([1,2,3,4,5])
print (type(a))
b=np.array([1,2,3,4,5])

print (a*b)


# In[48]:


a+1

