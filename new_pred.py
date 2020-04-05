#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
def pred_creation(pos_data_file,neg_data_file,polynomial_degree=0,trig_func=False,only_init=True,inequalities=False):
    names=[]
    data_x_pos = np.genfromtxt(pos_data_file, delimiter=' ')
    data_x_neg=  np.genfromtxt(neg_data_file, delimiter=' ')
    init_feat_num=data_x_pos.shape[1]
    for i in range(data_x_pos.shape[1]):
        names.append('x%d'%(i+1))

    if(polynomial_degree!=0 or polynomial_degree!=1):
        data_x_pos,data_x_neg,names=polynomial_pred(data_x_pos,data_x_neg,polynomial_degree,names)

    if(trig_func):
        data_x_pos,data_x_neg,names=trig_pred(data_x_pos,data_x_neg,only_init,init_feat_num,names)
    if(inequalities):
        data_x_pos,data_x_neg,names=pred_ineq(data_x_pos,data_x_neg,names)

    return data_x_pos,data_x_neg,names



# In[65]:


def polynomial_pred(data_x_pos,data_x_neg,polynomial_degree,names):
    init_feat_num=data_x_pos.shape[1]
    i=2
    while(i<=polynomial_degree):
        new_feature_pos=[]
        new_feature_neg=[]

        for index,row in enumerate (data_x_pos):
            new_feature_pos.append(np.power(row[0:init_feat_num],i))
        data_x_pos=np.append(data_x_pos,new_feature_pos,1)

        for index,row in enumerate (data_x_neg):
            new_feature_neg.append(np.power(row[0:init_feat_num],i))
        data_x_neg=np.append(data_x_neg,new_feature_neg,1)
        new_names=[]
        for name in names[0:init_feat_num]:
            new_names.append(name+"R%d"%(i))
        names.extend(new_names)
        i=i+1
    return data_x_pos,data_x_neg,names


# In[141]:


def trig_pred(data_x_pos,data_x_neg,only_init,init_feat_num,names):
    new_feature_pos=[]
    new_feature_neg=[]
    if(only_init):
        for index,row in enumerate (data_x_pos):
            new_feat= np.concatenate((np.cos(row[0:init_feat_num]),np.sin(row[0:init_feat_num])),axis=None)
            new_feature_pos.append(np.concatenate((new_feat,np.tan(row[0:init_feat_num])),axis=None))
        data_x_pos=np.append(data_x_pos,new_feature_pos,1)
        for index,row in enumerate (data_x_neg):
            new_feat= np.concatenate((np.cos(row[0:init_feat_num]),np.sin(row[0:init_feat_num])),axis=None)
            new_feature_neg.append(np.concatenate((new_feat,np.tan(row[0:init_feat_num])),axis=None))
        data_x_neg=np.append(data_x_neg,new_feature_neg,1)

        new_names=[]
        for name in names[0:init_feat_num]:
            new_names.append('cos('+name+')')
            new_names.append('sin('+name+')')
            new_names.append('tan('+name+')')
        names.extend(new_names)

    else:
        for index,row in enumerate (data_x_pos):
            new_feat= np.concatenate((np.cos(row),np.sin(row)),axis=None)
            new_feature_pos.append(np.concatenate((new_feat,np.tan(row)),axis=None).tolist())
        data_x_pos=np.append(data_x_pos,new_feature_pos,1)

        for index,row in enumerate (data_x_neg):
            new_feat= np.concatenate((np.cos(row),np.sin(row)),axis=None)
            new_feature_neg.append(np.concatenate((new_feat,np.tan(row)),axis=None).tolist())
        data_x_neg=np.append(data_x_neg,new_feature_neg,1)

        new_names=[]
        for name in names:
            new_names.append('cos('+name+')')
            new_names.append('sin('+name+')')
            new_names.append('tan('+name+')')

        names.extend(new_names)

    return data_x_pos,data_x_neg,names


# In[120]:


def pred_ineq(data_x_pos,data_x_neg,names):
    new_feature_pos=[]
    new_feature_neg=[]
    for index,row in enumerate(data_x_pos):
        i=0;
        new_feat_row=[]
        while(i<row.size-1):
            curr=row[i]
            for x in row[i+1:]:
                new_feat_row.append(curr-x)
            i=i+1
        new_feature_pos.append(new_feat_row)
    data_x_pos=np.append(data_x_pos,new_feature_pos,1)

    for index,row in enumerate(data_x_neg):
        i=0;
        new_feat_row=[]
        while(i<row.size-1):
            curr=row[i]
            for x in row[i+1:]:
                new_feat_row.append(curr-x)
            i=i+1
        new_feature_neg.append(new_feat_row)

    data_x_neg=np.append(data_x_neg,new_feature_neg,1)

    i=0;
    new_names=[]
    while(i<len(names)-1):
        curr=names[i]
        for x in names[i+1:]:
            new_names.append(curr+"Ineq"+x)
        i=i+1
    names.extend(new_names)

    return data_x_pos,data_x_neg,names


# In[157]:


def spacing(data_x_pos,data_x_neg,k):
    data = np.concatenate((data_x_pos,data_x_neg),axis=0)
    feats_max = np.max(data,axis=0)
    feats_min = np.min(data,axis=0)
    feats_intervals=[]
    for index,val in enumerate (feats_max):
        feats_intervals.append(np.linspace(feats_min[index],val, num=k))
    return feats_intervals
