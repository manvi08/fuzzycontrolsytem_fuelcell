#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=np.array([[1,0.76,1.9,0.5],[1,0.76,1.8,0.7],[1,0.76,1.2,0.9],
              [2.01,0.06,-0.4,1.5],[1.02,1.76,0.9,-0.5],[1.1,0.55,-1.9,0.6],
              [2.02,0.14,0.16,0.5],[1.03,1.27,0.19,0.4],[1.2,0.725,1.09,0.7],
              [2.03,0.24,-0.15,1.9],[1.07,1.37,1.09,-0.3],[1.3,0.78,-1.8,0.8],
              [2.04,0.34,0.06,0.54],[1.05,1.47,0.29,0.2],[1.4,0.80,1.7,0.9],
              [2.05,0.44,-1.25,0.56],[1.06,0.66,0.39,-0.1],[1.5,0.81,-1.6,1],
              [2.07,0.54,0.66,0.3],[1.04,0.64,0.49,0.05],[1.6,0.82,1.5,1.1],
              [2.06,0.64,-0.23,0.2],[1.03,0.6,0.59,-0.06],[1.7,0.83,-1.4,1.2],
              [2.08,0.74,1.45,0.1],[1.04,0.706,0.69,0.07],[1.8,0.84,1.3,1.3],
              [2.09,0.84,-1.29,0.5],[1.01,0.726,0.79,-0.08],[1.9,0.85,-1.2,1.4],
              [2.11,0.94,1.39,1.5],[1.08,0.766,0.89,0.09],[1.10,0.86,1.1,1.5],
              [2.12,0.73,-1.19,2.5],[1.09,0.786,0.99,-0.15],[1.11,0.87,-1,1.6],
              [2.13,0.63,1.69,1.45],[1.17,0.776,0.1,0.09],[1.12,0.88,0.39,1.7],
              [2.14,0.53,-1.99,1.25],[1.18,0.716,1.2,-0.08],[1.13,0.89,-0.8,1.8],
              [2.15,0.43,0.11,1.15],[1.19,0.796,1.9,0.07],[1.14,0.77,0.7,-1.9],
              [2.16,0.33,-0.12,1.05],[1.20,0.976,1.3,-0.54],[1.15,0.79,-0.6,2],
              [2.17,0.23,0.13,2],[1.21,1.65,1.4,0.44],[1.16,0.90,0.5,1.9],
              [2.18,0.13,-2,0],[1.22,1.4,1.6,-0.34],[2,0.91,-0.4,2.1],
              [2.19,0.93,1,0.6],[1.22,1.2,1.5,0.24],[2.1,0.92,0.3,1.95],
              [2.21,0.62,-1.6,1.5],[1.23,3.07,1.7,-0.14],[2.2,0.93,-0.2,1.85],
              [2.22,0.72,1.5,1.4],[1.24,0.07,1.8,0.41],[2.3,0.94,0.1,1.75],
              [2.23,0.82,-1.8,1.3],[1.25,0.16,2.09,-1.05],[2.4,0.95,-0.15,1.65],
              [2.24,0.62,0.5,1.2],[1.26,0.26,0.2,1.16],[2.5,0.96,2,1.55],
              [2.25,0.52,-0.6,1.1],[1.27,0.36,2.2,-1.24],[2.6,0.97,-2.01,1.45],
              [2.26,0.42,0.7,1],[1.28,0.46,2.06,1.36],[2.7,0.98,1.95,1.35],
              [2.27,0.32,-0.8,0.8],[1.29,0.56,1.56,-1.47],[2.8,0.99,-1.92,1.25],
              [2.28,0.22,0.9,0.7],[1.30,0.69,1.79,1.58],[2.9,1.00,1.05,1.15],
              [2.29,0.12,-1.5,0.6],[1.31,0.61,1.99,-1.66],[2.06,0.50,-0.95,1.05],
              [2.31,0.11,1.3,0.5],[1.32,0.62,1.29,1.75],[2.08,0.51,0.84,0.95],
              [2.32,0.71,-2.2,0.3],[1.33,0.63,1.49,-1.86],[2.45,0.52,-0.82,0.85],
              [2.33,0.81,0.9,1.5],[1.35,0.64,1.79,-1.99],[2.82,0.53,0.19,0.75],
              [2.34,0.91,-0.8,-0.3],[1.36,0.65,1.89,-2],[2.75,0.54,-0.05,0.65],
              [2.35,1.03,0.7,0.5],[1.37,0.68,1.39,0.25],[2.66,0.55,0.06,0.55],
              [2.36,1.09,-0.4,-1],[1.38,0.90,1.19,-0.95],[2.41,0.56,-0.02,0.45]])
dataset=pd.DataFrame(data, index=range(102),columns=["voltage","Hydroconc","error","coferror"])
h=[1,-1,2,-2,0.5,-0.5,0.6,-0.6,0.7,0.8,0.9,-0.9
              ,-0.8,0.11,0.12,0.13,-0.14,0.15,-0.13,0.14,-0.15,0.16,0.17,0.18,-0.16,-0.18,-0.17,0.19,0.19,0.2
              ,0.21,0.22,0.23,-0.21,-0.22,-0.2,-0.23,0.24,0.25,0.27,-0.27,0.26,-0.26,0.3,0.31,0.34,-0.54,0.55,1.1,1.2,1.3
              ,-1.1,-1.2,-1.3,1.4,1.5,1.6,-1.4,-1.5,-1.6,1.7,1.8,1.9,
 -1.9,-1.8,-1.7,2,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,
 0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.75,0.78,0.76,0.77,0.79,-0.8,0.81,
 -0.82,0.83,0.84,0.85,0.89,0.9,-0.91,0.92,0.93]
dataset.insert(4,"Hydroflw",h,True)


# In[2]:


dataset.to_csv('one.csv')


# In[5]:


import skfuzzy as fuzz
from skfuzzy import control as ctrl
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
error= ctrl.Antecedent(np.arange(-2,3,0.1),'error(x10^-4)')
null=ctrl.Consequent(np.arange(0,2,1),'null')
null.automf(3)
sig=0.5
error["NB"]=fuzz.gaussmf(error.universe, -2,sig)
error["NM"]=fuzz.gaussmf(error.universe, -1,sig)
error["ZO"]=fuzz.gaussmf(error.universe, 0,sig)
error["PM"]=fuzz.gaussmf(error.universe, 1,sig)
error["PB"]=fuzz.gaussmf(error.universe, 2,sig)
error.view()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
coferror= ctrl.Antecedent(np.arange(-2,3,0.1),'coferror(x10^-4)')
null=ctrl.Consequent(np.arange(0,2,1),'null')
null.automf(3)
sig=0.5
coferror["NB"]=fuzz.gaussmf(coferror.universe, -2,sig)
coferror["NM"]=fuzz.gaussmf(coferror.universe, -1,sig)
coferror["ZO"]=fuzz.gaussmf(coferror.universe, 0,sig)
coferror["PM"]=fuzz.gaussmf(coferror.universe, 1,sig)
coferror["PB"]=fuzz.gaussmf(coferror.universe, 2,sig)
coferror.view()


# In[7]:


universe=np.linspace(-2,2,5)
error=ctrl.Antecedent(universe,'error')
coferror=ctrl.Antecedent(universe,'coferror')
Hydroflw=ctrl.Consequent(universe,'Hydroflw')
names=['NB','NM','ZO','PM','PB']
error.automf(names=names)
coferror.automf(names=names)
Hydroflw.automf(names=names)


# In[38]:


rule0 = ctrl.Rule(antecedent=((error['NB'] & coferror['NB']) |
                              (error['NM'] & coferror['NB']) |
                              (error['NB'] & coferror['NM'])),
                  consequent=Hydroflw['NB'], label='rule NB')

rule1 = ctrl.Rule(antecedent=((error['NB'] & coferror['ZO']) |
                              (error['NB'] & coferror['PM']) |
                              (error['NM'] & coferror['NM']) |
                              (error['NM'] & coferror['ZO']) |
                              (error['ZO'] & coferror['NM']) |
                              (error['ZO'] & coferror['NB']) |
                              (error['PM'] & coferror['NB'])),
                  consequent=Hydroflw['NM'], label='rule NM')

rule2 = ctrl.Rule(antecedent=((error['NB'] & coferror['PB']) |
                              (error['NM'] & coferror['PM']) |
                              (error['ZO'] & coferror['ZO']) |
                              (error['PM'] & coferror['NM']) |
                              (error['PB'] & coferror['NB'])),
                  consequent=Hydroflw['ZO'], label='rule ZO')

rule3 = ctrl.Rule(antecedent=((error['NM'] & coferror['PB']) |
                              (error['ZO'] & coferror['PB']) |
                              (error['ZO'] & coferror['PM']) |
                              (error['PM'] & coferror['PM']) |
                              (error['PM'] & coferror['ZO']) |
                              (error['PB'] & coferror['ZO']) |
                              (error['PB'] & coferror['NM'])),
                  consequent=Hydroflw['PM'], label='rule PM')

rule4 = ctrl.Rule(antecedent=((error['PM'] & coferror['PB']) |
                              (error['PB'] & coferror['PB']) |
                              (error['PB'] & coferror['PM'])),
                  consequent=Hydroflw['PB'], label='rule PB')
rule1.view()


# In[9]:


system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4])
sim = ctrl.ControlSystemSimulation(system, flush_after_run=21 * 21 + 1)


# In[10]:


upsampled=np.linspace(-2,2,21)
x,y=np.meshgrid(upsampled,upsampled)
z=np.zeros_like(x)
for i in range(21):
    for j in range(21):
        sim.input['error']=x[i,j]
        sim.input['coferror']=y[i,j]
        sim.compute()
        z[i,j]=sim.output['Hydroflw']
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
cset=ax.contourf(x,y,z,zdir='z',offset=-2.5,cmap='viridis',alpha=0.5)
cset=ax.contourf(x,y,z,zdir='x',offset=3,cmap='viridis',alpha=0.5)
cset=ax.contourf(x,y,z,zdir='y',offset=3,cmap='viridis',alpha=0.5)
ax.view_init(30,200)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'notebook')
Hydroflw= ctrl.Antecedent(np.arange(-2,3,0.0001),'Hydroflw(x10^-2)')
null=ctrl.Consequent(np.arange(0,2,1),'null')
null.automf(5)
sig=0.5
Hydroflw["NB"]=fuzz.gaussmf(Hydroflw.universe, -2,sig)
Hydroflw["NM"]=fuzz.gaussmf(Hydroflw.universe, -1,sig)
Hydroflw["ZO"]=fuzz.gaussmf(Hydroflw.universe, 0,sig)
Hydroflw["PM"]=fuzz.gaussmf(Hydroflw.universe, 1,sig)
Hydroflw["PB"]=fuzz.gaussmf(Hydroflw.universe, 2,sig)
Hydroflw.view()


# In[15]:


hf_ctrl=ctrl.ControlSystemSimulation(system)


# In[36]:


e=float(input())
ce=float(input())
hf_ctrl.input['error']=e
hf_ctrl.input['coferror']=ce
hf_ctrl.compute()


# In[37]:


print(hf_ctrl.output['Hydroflw'])
if(hf_ctrl.output['Hydroflw']>=1.5): Hydroflw['PB'].view()
if(hf_ctrl.output['Hydroflw']>=0.5 and hf_ctrl.output['Hydroflw']<1.5): Hydroflw['PM'].view()
if(hf_ctrl.output['Hydroflw']>=0 and hf_ctrl.output['Hydroflw']<0.5): Hydroflw['ZO'].view()
if(hf_ctrl.output['Hydroflw']>=(-1.5) and hf_ctrl.output['Hydroflw']<0): Hydroflw['NM'].view()
if(hf_ctrl.output['Hydroflw']<(-1.5)): Hydroflw['NB'].view()
#Hydroflw.view(sim=hf_ctrl)


# In[ ]:




