#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns
import pandas as pd

from collections import OrderedDict


# In[2]:


def pretty_dataset_name(dataset_name):
    if dataset_name == 'eth':
        return 'ETH - Univ'
    elif dataset_name == 'hotel':
        return 'ETH - Hotel'
    elif dataset_name == 'univ':
        return 'UCY - Univ'
    elif dataset_name == 'zara1':
        return 'UCY - Zara 1'
    elif dataset_name == 'zara2':
        return 'UCY - Zara 2'
    else:
        return dataset_name


# In[3]:


dataset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'Average']
alg_name = "Ours"


# # Displacement Error Analysis

# In[79]:



# These are for a prediction horizon of 12 timesteps.
prior_work_ade_results = {
}

linestyles = ['--', '-.', '-', ':']


# In[81]:


mean_markers = 'X'
marker_size = 7
line_colors = ['#1f78b4','#33a02c','#fb9a99','#e31a1c']
area_colors = ['#80CBE5','#ABCB51', '#F05F78']
area_rgbs = list()
for c in area_colors:
    area_rgbs.append([int(c[i:i+2], 16) for i in (1, 3, 5)])


# In[94]:


# Load Ours
perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}*attention_radius_3*fde_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']
perf_df = perf_df.rename(columns={"metric": "error_type", "value": "error_value"})


# In[95]:


# Load Trajectron and GAN
errors_df = pd.concat([pd.read_csv(f) for f in glob.glob('csv/old/curr_*_errors.csv')], ignore_index=True)
del errors_df['data_precondition']
errors_df = errors_df[~(errors_df['method'] == 'our_full')]
errors_df = errors_df[~(errors_df['error_type'] == 'mse')]
errors_df.loc[errors_df['error_type'] =='fse', 'error_type'] = 'fde'
#errors_df.loc[errors_df['error_type'] =='mse', 'error_type'] = 'ade'
errors_df.loc[errors_df['method'] == 'our_most_likely', 'method'] = 'Trajectron'


# In[96]:


perf_df = perf_df.append(errors_df)
errors_df


# In[97]:


with sns.color_palette("muted"):
    fig_fse, ax_fses = plt.subplots(nrows=1, ncols=6, figsize=(8, 4), dpi=300, sharey=True)
    for idx, ax_fse in enumerate(ax_fses):
        dataset_name = dataset_names[idx]
        if dataset_name != 'Average':
            specific_df = perf_df[(perf_df['dataset'] == dataset_name) & (perf_df['error_type'] == 'fde')]
            specific_df['dataset'] = pretty_dataset_name(dataset_name)
        else:
            specific_df = perf_df[(perf_df['error_type'] == 'fde')].copy()
            specific_df['dataset'] = 'Average'

        sns.boxplot(x='dataset', y='error_value', hue='method',
            data=specific_df, ax=ax_fse, showfliers=False,
            palette=area_colors, hue_order=['sgan', 'Trajectron', alg_name], width=2.)
        
        ax_fse.get_legend().remove()
        ax_fse.set_xlabel('')
        ax_fse.set_ylabel('' if idx > 0 else 'Final Displacement Error (m)')

        ax_fse.scatter([-0.665, 0, 0.665],
               [np.mean(specific_df[specific_df['method'] == 'sgan']['error_value']),
                np.mean(specific_df[specific_df['method'] == 'Trajectron']['error_value']),
                np.mean(specific_df[specific_df['method'] == alg_name]['error_value'])],
               s=marker_size*marker_size, c=np.asarray(area_rgbs)/255.0, marker=mean_markers,
               edgecolors='#545454', zorder=10)
        
        ax_fse.axhline(y=fse_val, label=baseline, color=line_colors[baseline_idx], linestyle=linestyles[baseline_idx])
            
        if idx == 0:
            handles, labels = ax_fse.get_legend_handles_labels()


            handles = [handles[0], handles[4], handles[1], handles[5], handles[2], handles[6], handles[3]]
            labels = [labels[0], 'Social GAN', labels[1], 'Trajectron', labels[2], alg_name, labels[3]]

            ax_fse.legend(handles, labels, 
                          loc='lower center', bbox_to_anchor=(0.5, 0.9),
                          ncol=4, borderaxespad=0, frameon=False,
                          bbox_transform=fig_fse.transFigure)


#     fig_fse.text(0.51, 0.03, 'Dataset', ha='center')

plt.savefig('plots/fde_boxplots.pdf', dpi=300, bbox_inches='tight')


# In[98]:


del perf_df
del errors_df


# # Average Displacement Error

# In[99]:


# Load Ours
perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}*attention_radius_3*ade_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']
perf_df = perf_df.rename(columns={"metric": "error_type", "value": "error_value"})
#perf_df.head()


# In[100]:


# Load Trajectron and GAN
errors_df = pd.concat([pd.read_csv(f) for f in glob.glob('old/curr_*_errors.csv')], ignore_index=True)
del errors_df['data_precondition']
errors_df = errors_df[~(errors_df['method'] == 'our_full')]
errors_df = errors_df[~(errors_df['error_type'] == 'fse')]
#errors_df.loc[errors_df['error_type'] =='fse', 'error_type'] = 'fde'
errors_df.loc[errors_df['error_type'] =='mse', 'error_type'] = 'ade'
errors_df.loc[errors_df['method'] == 'our_most_likely', 'method'] = 'Trajectron'


# In[101]:


perf_df = perf_df.append(errors_df)
del errors_df


# In[102]:


with sns.color_palette("muted"):
    fig_fse, ax_fses = plt.subplots(nrows=1, ncols=6, figsize=(8, 4), dpi=300, sharey=True)
    for idx, ax_fse in enumerate(ax_fses):
        dataset_name = dataset_names[idx]
        if dataset_name != 'Average':
            specific_df = perf_df[(perf_df['dataset'] == dataset_name) & (perf_df['error_type'] == 'ade')]
            specific_df['dataset'] = pretty_dataset_name(dataset_name)
        else:
            specific_df = perf_df[(perf_df['error_type'] == 'ade')].copy()
            specific_df['dataset'] = 'Average'

        sns.boxplot(x='dataset', y='error_value', hue='method',
            data=specific_df, ax=ax_fse, showfliers=False,
            palette=area_colors, hue_order=['sgan', 'Trajectron', alg_name], width=2.)

        ax_fse.get_legend().remove()
        ax_fse.set_xlabel('')
        ax_fse.set_ylabel('' if idx > 0 else 'Average Displacement Error (m)')

        ax_fse.scatter([-0.665, 0, 0.665],
               [np.mean(specific_df[specific_df['method'] == 'sgan']['error_value']),
                np.mean(specific_df[specific_df['method'] == 'Trajectron']['error_value']),
                np.mean(specific_df[specific_df['method'] == alg_name]['error_value'])],
               s=marker_size*marker_size, c=np.asarray(area_rgbs)/255.0, marker=mean_markers,
               edgecolors='#545454', zorder=10)
        
        for baseline_idx, (baseline, fse_val) in enumerate(prior_work_ade_results[pretty_dataset_name(dataset_name)].items()):
            ax_fse.axhline(y=fse_val, label=baseline, color=line_colors[baseline_idx], linestyle=linestyles[baseline_idx])
            
        if idx == 0:
            handles, labels = ax_fse.get_legend_handles_labels()


            handles = [handles[0], handles[4], handles[1], handles[5], handles[2], handles[6], handles[3]]
            labels = [labels[0], 'Social GAN', labels[1], 'Trajectron', labels[2], alg_name, labels[3]]

            ax_fse.legend(handles, labels, 
                          loc='lower center', bbox_to_anchor=(0.5, 0.9),
                          ncol=4, borderaxespad=0, frameon=False,
                          bbox_transform=fig_fse.transFigure)

#     fig_fse.text(0.51, 0.03, 'Dataset', ha='center')

plt.savefig('plots/ade_boxplots.pdf', dpi=300, bbox_inches='tight')


# In[12]:


del perf_df


# # KDE Negative Log Likelihood Attention Radius 3m

# In[4]:


# Load Ours
perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_12*kde_full.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True)
        del perf_df['Unnamed: 0']
#perf_df.head()


# In[5]:


# # Load Trajectron and SGAN
# lls_df = pd.concat([pd.read_csv(f) for f in glob.glob('csv/old/curr_*_lls.csv')], ignore_index=True)
# lls_df.loc[lls_df['method'] == 'our_full', 'method'] = 'Trajectron'
# lls_df['error_type'] = 'KDE'
# #lls_df.head()


# In[6]:


for dataset in dataset_names:
    if dataset != 'Average':
        print('KDE NLL for ' + pretty_dataset_name(dataset))
        #print(f"SGAN: {-lls_df[(lls_df['method'] == 'sgan') & (lls_df['dataset'] == dataset)]['log-likelihood'].mean()}")
        #print(f"Trajectron: {-lls_df[(lls_df['method'] == 'Trajectron')  & (lls_df['dataset'] == dataset)]['log-likelihood'].mean()}")
        print(f"{alg_name}: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print('KDE NLL for ' + pretty_dataset_name(dataset))
        #print(f"SGAN: {-lls_df[(lls_df['method'] == 'sgan')]['log-likelihood'].mean()}")
        #print(f"Trajectron: {-lls_df[(lls_df['method'] == 'Trajectron')]['log-likelihood'].mean()}")
        print(f"{alg_name}: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")
              


# In[7]:


del perf_df


# # Most Likely FDE Attention Radius 3m

# In[8]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_12*fde_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = 'Trajectron++'
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[9]:


for dataset in dataset_names:
    print('FDE Most Likely for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++') & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++')]['value'].mean()}")


# In[10]:


del perf_df


# # Most Likely ADE Attention Radius 3m

# In[11]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_12*ade_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = 'Trajectron++'
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[12]:


for dataset in dataset_names:
    print('ADE Most Likely for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++') & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++')]['value'].mean()}")


# In[13]:


del perf_df


# # Best of 20 Evaluation FDE Attention Radius 3m

# In[14]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_12*fde_best_of.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[15]:


for dataset in dataset_names:
    print('FDE Best of 20 for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")


# In[16]:


del perf_df


# # Best of 20 Evaluation ADE Attention Radius 3m

# In[17]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_12*ade_best_of.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[18]:


for dataset in dataset_names:
    print('ADE Best of 20 for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")


# In[19]:


del perf_df


# # KDE Negative Log Likelihood Attention Radius 3m Velocity

# In[20]:


# Load Ours
perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_vel_12*kde_full.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True)
        del perf_df['Unnamed: 0']
#perf_df.head()


# In[21]:


# # Load Trajectron and SGAN
# lls_df = pd.concat([pd.read_csv(f) for f in glob.glob('csv/old/curr_*_lls.csv')], ignore_index=True)
# lls_df.loc[lls_df['method'] == 'our_full', 'method'] = 'Trajectron'
# lls_df['error_type'] = 'KDE'
# #lls_df.head()


# In[22]:


for dataset in dataset_names:
    if dataset != 'Average':
        print('KDE NLL for ' + pretty_dataset_name(dataset))
        #print(f"SGAN: {-lls_df[(lls_df['method'] == 'sgan') & (lls_df['dataset'] == dataset)]['log-likelihood'].mean()}")
        #print(f"Trajectron: {-lls_df[(lls_df['method'] == 'Trajectron')  & (lls_df['dataset'] == dataset)]['log-likelihood'].mean()}")
        print(f"{alg_name}: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print('KDE NLL for ' + pretty_dataset_name(dataset))
        #print(f"SGAN: {-lls_df[(lls_df['method'] == 'sgan')]['log-likelihood'].mean()}")
        #print(f"Trajectron: {-lls_df[(lls_df['method'] == 'Trajectron')]['log-likelihood'].mean()}")
        print(f"{alg_name}: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")
              


# In[23]:


del perf_df


# # Most Likely FDE Attention Radius 3m Velocity

# In[24]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_vel_12*fde_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = 'Trajectron++'
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[25]:


for dataset in dataset_names:
    print('FDE Most Likely for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++') & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++')]['value'].mean()}")


# In[26]:


del perf_df


# # Most Likely Evaluation ADE Attention Radius 3m Velocity

# In[27]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_vel_12*ade_most_likely.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = 'Trajectron++'
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[28]:


for dataset in dataset_names:
    print('ADE Most Likely for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++') & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"{alg_name}: {perf_df[(perf_df['method'] == 'Trajectron++')]['value'].mean()}")


# In[29]:


del perf_df


# # Best of 20 Evaluation FDE Attention Radius 3m Velocity

# In[30]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_vel_12*fde_best_of.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[31]:


for dataset in dataset_names:
    print('FDE Best of 20 for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")


# In[32]:


del perf_df


# # Best of 20 Evaluation ADE Attention Radius 3m Velocity

# In[33]:


perf_df = pd.DataFrame()
for dataset in dataset_names:
    for f in glob.glob(f"results/{dataset}_vel_12*ade_best_of.csv"):
        print(f)
        dataset_df = pd.read_csv(f)
        dataset_df['dataset'] = dataset
        dataset_df['method'] = alg_name
        perf_df = perf_df.append(dataset_df, ignore_index=True, sort=False)
        del perf_df['Unnamed: 0']


# In[34]:


for dataset in dataset_names:
    print('ADE Best of 20 for ' + pretty_dataset_name(dataset))
    if dataset != 'Average':
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name) & (perf_df['dataset'] == dataset)]['value'].mean()}")
    else:
        print(f"Trajectron++: {perf_df[(perf_df['method'] == alg_name)]['value'].mean()}")


# In[35]:


del perf_df


# In[ ]:




