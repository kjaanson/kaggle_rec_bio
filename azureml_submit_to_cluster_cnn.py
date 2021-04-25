#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core import Experiment
from azureml.core import Workspace, Run
from azureml.core import Environment
from azureml.core import Dataset, Datastore

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig


# In[2]:


from azureml.core.compute import ComputeInstance


# In[3]:


workspace = Workspace.from_config()


# In[4]:


instance = ComputeTarget(workspace=workspace, name='gpu-compute-low')


# In[5]:


dataset = Dataset.get_by_name(workspace, name='recursionbio')


# In[6]:


dataset


# In[7]:


env_list = Environment.list(workspace)


# In[8]:


env_list.keys()


# In[9]:


tf_env = Environment.get(workspace=workspace, name='AzureML-TensorFlow-2.3-GPU')
tf_env = tf_env.clone(new_name='recbio-tf-2.3-efficientnet')


# In[10]:


tf_env.python.conda_dependencies.add_conda_package('scikit-learn')
tf_env.python.conda_dependencies.add_conda_package('scipy')
tf_env.python.conda_dependencies.add_conda_package('matplotlib')


# In[11]:


tf_env.python.conda_dependencies.add_pip_package('horovod==0.19.5')
tf_env.python.conda_dependencies.add_pip_package('retry')


# In[12]:


tf_env


# In[13]:


train_scr = ScriptRunConfig(
    source_directory='./scripts',
    script='train_cnn.py',
    arguments=['--data-path', dataset.as_mount(),
               '--epochs', 10,
               '--batch', 8,
               '--train-frac', 0.5],
    compute_target=instance,
    environment=tf_env
)


# In[14]:


train_scr


# In[15]:


run = Experiment(workspace=workspace, name='recbio-effnet-model').submit(train_scr)


# In[ ]:


run.wait_for_completion(show_output=True)


# In[ ]:





# In[ ]:




