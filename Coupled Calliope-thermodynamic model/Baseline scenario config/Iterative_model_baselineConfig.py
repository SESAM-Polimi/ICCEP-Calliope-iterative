# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019

A script to run and post-process the ICCEP model

@author: F.Lombardi
"""
#%% Initialisation
import calliope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

calliope.set_log_level('ERROR') #sets the level of verbosity of Calliope's operations

ring_heat_diff = np.ones(744)
T_ring_diff = np.ones(744)*1.1
ring_heat_iter = []
T_ring_iter = []

cp = 4.18 # cp of Water
Mw = 3000000 # mass of water in the ring
Tmax = 20 # defines maximum tempearature of the ring
Tmin = 10 # defines minumum tempearature of the ring
T_0 = 15 # defines ring temperature at step 0
T_dhw = 40 #defines lowest T of dhw_storage
T_sh = 25 #defines lowest T of sh_storage
T_co = 5 #defines lowest T of co_storage
C_dhw_r = 186.04 #kWh/K
C_sh_r = 372.09 #kWh/K
C_co_r = 372.09 #kWh/K
C_sh_c = 139.53 #kWh/K
C_co_c = 139.53 #kWh/K
#Q_fresh = (12558000) #kJ provided/extracted by groundwater pumps

#%% Iteration 0
'''
Iteration 0
'''

'''
Calliope
'''
model_iter_0 = calliope.Model('model_op.yaml', scenario='iteration_zero,no_TES')
model_iter_0.run()

hp_dhw_supply_0 = model_iter_0.get_formatted_array('carrier_prod').loc[{'techs':'hp_dhw','carriers':'dhw'}].to_pandas().T #electricity production per technology in kWh
hp_sh_supply_0 = model_iter_0.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_hot','carriers':'sh'}].to_pandas().T #electricity production per technology in kWh
hp_co_supply_0 = model_iter_0.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_cold','carriers':'air_cond'}].to_pandas().T #electricity production per technology in kWh
cop_dhw_trend_0 = pd.read_csv('timeseries_data/cop_dhw_start.csv', index_col=0)[str(hp_dhw_supply_0.index[0]):str(hp_dhw_supply_0.index[-1])]#.set_index('Unnamed: 0', inplace=True)
cop_sh_trend_0 = pd.read_csv('timeseries_data/cop_sh_start.csv', index_col=0)[str(hp_dhw_supply_0.index[0]):str(hp_dhw_supply_0.index[-1])]#.set_index('Unnamed: 0', inplace=True)
cop_co_trend_0 = pd.read_csv('timeseries_data/cop_co_start.csv', index_col=0)[str(hp_dhw_supply_0.index[0]):str(hp_dhw_supply_0.index[-1])]#.set_index('Unnamed: 0', inplace=True)
ring_heat_dhw_0 = -(hp_dhw_supply_0)*(1-1/cop_dhw_trend_0)
ring_heat_sh_0 = -(hp_sh_supply_0)*(1-1/cop_sh_trend_0)
ring_heat_co_0 = (hp_co_supply_0)*(1+1/cop_co_trend_0)
ring_heat_tot_0 = pd.DataFrame([ring_heat_dhw_0.values[:,0],ring_heat_sh_0.values.sum(axis=1),ring_heat_co_0.values.sum(axis=1)]).T
ring_heat_iter.append(ring_heat_tot_0.sum(axis=1).values)

'''
Thermodynamic model accounting for the DH_ring interaction with HPs
'''

COP_dhw_series_r = [] 
COP_sh_series_r = [] 
COP_co_series_r = [] 
COP_dhw_series_c = [] 
COP_sh_series_c = [] 
COP_co_series_c = [] 
T = []
T.append(T_0)

for i in range(0,len(hp_dhw_supply_0)):
       
    COP_dhw_series_r.append(6.81 - 0.121*(45-T_0) + 0.000630*math.pow((45-T_0),2))
    COP_sh_series_r.append(6.81 - 0.121*(35-T_0) + 0.000630*math.pow((35-T_0),2))
    COP_co_series_r.append(6.755 - 0.141*(13-T_0) + 0.003*math.pow((13-T_0),2))
    COP_sh_series_c.append(6.81 - 0.121*(35-T_0) + 0.000630*math.pow((35-T_0),2))
    COP_co_series_c.append(6.755 - 0.141*(13-T_0) + 0.003*math.pow((13-T_0),2))

COP_dhw_series_r = pd.DataFrame(COP_dhw_series_r)
COP_sh_series_r = pd.DataFrame(COP_sh_series_r)
COP_co_series_r = pd.DataFrame(COP_co_series_r)
COP_sh_series_c = pd.DataFrame(COP_sh_series_c)
COP_co_series_c = pd.DataFrame(COP_co_series_c)
T_ring_iter.append(T)

cop_dhw_new_trend_0 = pd.DataFrame([COP_dhw_series_r.values[:,0],COP_dhw_series_r.values[:,0]]).T.set_index(cop_dhw_trend_0.index)
cop_dhw_new_trend_0.columns = ['buildings','commercial']
cop_dhw_new_trend_0.to_csv('timeseries_data/cop_dhw.csv')

cop_sh_new_trend_0 = pd.DataFrame([COP_sh_series_r.values[:,0],COP_sh_series_c.values[:,0]]).T.set_index(cop_sh_trend_0.index)
cop_sh_new_trend_0.columns = ['buildings','commercial']
cop_sh_new_trend_0.to_csv('timeseries_data/cop_sh.csv')

cop_co_new_trend_0 = pd.DataFrame([COP_co_series_r.values[:,0],COP_co_series_c.values[:,0]]).T.set_index(cop_co_trend_0.index)
cop_co_new_trend_0.columns = ['buildings','commercial']
cop_co_new_trend_0.to_csv('timeseries_data/cop_co.csv')


#%% Iteration until convergence script
'''
Iterations until convergence
'''
j = 0
divergence_record = []
hp_dhw_rec= []
hp_sh_rec = []
hp_co_rec = []

while np.nanmean(abs(T_ring_diff)) > 0.1: #np.nanmean(abs(T_ring_diff[np.where(abs(T_ring_diff)!= float('inf'))])) > 0.05: #
    print('previous divergence: %f' %(float(np.nanmean(abs(T_ring_diff)))))
    print('iteration %d' % (j+1))
    j += 1
    
    '''
    Calliope
    '''
    model_iter = calliope.Model('model_op.yaml', scenario='no_TES')
    model_iter.run()
    
    hp_dhw_supply = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_dhw','carriers':'dhw'}].to_pandas().T #electricity production per technology in kWh
    hp_sh_supply = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_hot','carriers':'sh'}].to_pandas().T #electricity production per technology in kWh
    hp_co_supply = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_cold','carriers':'air_cond'}].to_pandas().T #electricity production per technology in kWh
    cop_dhw_trend = pd.read_csv('timeseries_data/cop_dhw.csv', index_col=0)[str(hp_dhw_supply.index[0]):str(hp_dhw_supply.index[-1])]#.set_index('Unnamed: 0', inplace=True)
    cop_sh_trend = pd.read_csv('timeseries_data/cop_sh.csv', index_col=0)[str(hp_dhw_supply.index[0]):str(hp_dhw_supply.index[-1])]#.set_index('Unnamed: 0', inplace=True)
    cop_co_trend = pd.read_csv('timeseries_data/cop_co.csv', index_col=0)[str(hp_dhw_supply.index[0]):str(hp_dhw_supply.index[-1])]#.set_index('Unnamed: 0', inplace=True)
    ring_heat_dhw = -(hp_dhw_supply)*(1-1/cop_dhw_trend)
    ring_heat_sh = -(hp_sh_supply)*(1-1/cop_sh_trend)
    ring_heat_co = (hp_co_supply)*(1+1/cop_co_trend)
    ring_heat_tot = pd.DataFrame([ring_heat_dhw.values[:,0],ring_heat_sh.values.sum(axis=1),ring_heat_co.values.sum(axis=1)]).T
    ring_heat_iter.append(ring_heat_tot.sum(axis=1).values)
    
    '''
    Thermodynamic model accounting for the DH_ring interaction with HPs
    '''

    COP_dhw_series_r = [] 
    COP_sh_series_r = [] 
    COP_co_series_r = [] 
    COP_dhw_series_c = [] 
    COP_sh_series_c = [] 
    COP_co_series_c = [] 
    T = []
    T.append(T_0)

    for i in range(0,len(hp_dhw_supply)):
          
        COP_dhw_series_r.append(6.81 - 0.121*(45-T_0) + 0.000630*math.pow((45-T_0),2))
        COP_sh_series_r.append(6.81 - 0.121*(35-T_0) + 0.000630*math.pow((35-T_0),2))
        COP_co_series_r.append(6.755 - 0.141*(13-T_0) + 0.003*math.pow((13-T_0),2))
        COP_sh_series_c.append(6.81 - 0.121*(35-T_0) + 0.000630*math.pow((35-T_0),2))
        COP_co_series_c.append(6.755 - 0.141*(13-T_0) + 0.003*math.pow((13-T_0),2))
    
    COP_dhw_series_r = pd.DataFrame(COP_dhw_series_r)
    COP_sh_series_r = pd.DataFrame(COP_sh_series_r)
    COP_co_series_r = pd.DataFrame(COP_co_series_r)
    COP_sh_series_c = pd.DataFrame(COP_sh_series_c)
    COP_co_series_c = pd.DataFrame(COP_co_series_c)
    T_ring_iter.append(T)
    
    cop_dhw_new_trend = pd.DataFrame([COP_dhw_series_r.values[:,0],COP_dhw_series_r.values[:,0]]).T.set_index(cop_dhw_trend_0.index)
    cop_dhw_new_trend.columns = ['buildings','commercial']
    cop_dhw_new_trend.to_csv('timeseries_data/cop_dhw.csv')
    
    cop_sh_new_trend = pd.DataFrame([COP_sh_series_r.values[:,0],COP_sh_series_c.values[:,0]]).T.set_index(cop_sh_trend_0.index)
    cop_sh_new_trend.columns = ['buildings','commercial']
    cop_sh_new_trend.to_csv('timeseries_data/cop_sh.csv')
    
    cop_co_new_trend = pd.DataFrame([COP_co_series_r.values[:,0],COP_co_series_c.values[:,0]]).T.set_index(cop_co_trend_0.index)
    cop_co_new_trend.columns = ['buildings','commercial']
    cop_co_new_trend.to_csv('timeseries_data/cop_co.csv')   
    
    '''
    Convergence check
    '''
    ring_heat_diff = (ring_heat_iter[j]-ring_heat_iter[j-1])/ring_heat_iter[j]
    T_ring_diff = (np.array(T_ring_iter[j])-np.array(T_ring_iter[j-1]))/np.array(T_ring_iter[j-1])
    #T_ring_diff = (np.array(T_ring_iter[j])-np.array(T_ring_iter[j-1]))
    model_iter.to_netcdf('results_%d' % (j+1))
    divergence_record.append(np.nanmean(abs(T_ring_diff)))
    pd.DataFrame(divergence_record).to_csv('div_record.csv')
    hp_dhw_rec.append(hp_dhw_supply)
    hp_sh_rec.append(hp_sh_supply)
    hp_co_rec.append(hp_co_supply)

#%% Plot T oscillations
T_ring_iter_df = pd.DataFrame(T_ring_iter).T

time = np.linspace(0,len(hp_dhw_supply_0),len(hp_dhw_supply_0))
fig = plt.figure(figsize=(15,10))
for i in range(10,len(ring_heat_iter)):
    plt.plot(T_ring_iter_df[i][1:])
    plt.legend()

#%% Aggregate plots pre-processing
demand_dhw = model_iter.get_formatted_array('carrier_con').loc[{'techs':'demand_dhw', 'carriers':'dhw'}].sum('locs').to_pandas().T
demand_sh = model_iter.get_formatted_array('carrier_con').loc[{'techs':'demand_sh', 'carriers':'sh'}].sum('locs').to_pandas().T
demand_co = model_iter.get_formatted_array('carrier_con').loc[{'techs':'demand_co', 'carriers':'air_cond'}].sum('locs').to_pandas().T
demand_el = model_iter.get_formatted_array('carrier_con').loc[{'techs':'demand_el', 'carriers':'electricity'}].sum('locs').to_pandas().T

hps_dhw = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_dhw', 'carriers':'dhw'}].sum('locs').to_pandas().T
hps_dhw_inputs = model_iter.get_formatted_array('carrier_con').loc[{'techs':'hp_dhw', 'carriers':'electricity'}].sum('locs').to_pandas().T
#tes_dhw_out = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'tes_dhw', 'carriers':'dhw'}].sum('locs').to_pandas().T
#tes_dhw_in = model_iter.get_formatted_array('carrier_con').loc[{'techs':'tes_dhw', 'carriers':'dhw'}].sum('locs').to_pandas().T
#tes_dhw_o = tes_dhw_out+tes_dhw_in
#tes_dhw_o[tes_dhw_o<0]=0
#tes_dhw_i = tes_dhw_out+tes_dhw_in
#tes_dhw_i[tes_dhw_i>0]=0
#tes_dhw_cap = model_iter.get_formatted_array('storage').loc[{'techs':'tes_dhw'}].sum('locs').to_pandas().T

hps_sh = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_hot', 'carriers':'sh'}].sum('locs').to_pandas().T
hps_sh_inputs = model_iter.get_formatted_array('carrier_con').loc[{'techs':'hp_clim_hot', 'carriers':'electricity'}].sum('locs').to_pandas().T
#tes_sh_out = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'tes_sh', 'carriers':'sh'}].sum('locs').to_pandas().T
#tes_sh_in = model_iter.get_formatted_array('carrier_con').loc[{'techs':'tes_sh', 'carriers':'sh'}].sum('locs').to_pandas().T
#tes_sh_o = tes_sh_out+tes_sh_in
#tes_sh_o[tes_sh_o<0]=0
#tes_sh_i = tes_sh_out+tes_sh_in
#tes_sh_i[tes_sh_i>0]=0
#tes_sh_cap = model_iter.get_formatted_array('storage').loc[{'techs':'tes_sh'}].sum('locs').to_pandas().T

hps_co = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'hp_clim_cold', 'carriers':'air_cond'}].sum('locs').to_pandas().T
hps_co_inputs = model_iter.get_formatted_array('carrier_con').loc[{'techs':'hp_clim_cold', 'carriers':'electricity'}].sum('locs').to_pandas().T
#tes_co_out = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'tes_co', 'carriers':'air_cond'}].sum('locs').to_pandas().T
#tes_co_in = model_iter.get_formatted_array('carrier_con').loc[{'techs':'tes_co', 'carriers':'air_cond'}].sum('locs').to_pandas().T
#tes_co_o = tes_co_out+tes_co_in
#tes_co_o[tes_co_o<0]=0
#tes_co_i = tes_co_out+tes_co_in
#tes_co_i[tes_co_i>0]=0
#tes_co_cap = model_iter.get_formatted_array('storage').loc[{'techs':'tes_co'}].sum('locs').to_pandas().T

grid = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'grid', 'carriers':'electricity'}].sum('locs').to_pandas().T
pv_rooftop = model_iter.get_formatted_array('carrier_prod').loc[{'techs':'pv_rooftop', 'carriers':'electricity'}].sum('locs').to_pandas().T

#DHW production
hp_dhw = hps_dhw
#sto_dhw_o = hp_dhw + (tes_dhw_o)
hp_dhw_el = -hps_dhw_inputs
#sto_dhw_i = tes_dhw_i
loa_dhw = -demand_dhw

#SH production
hp_sh = hps_sh
#sto_sh_o = hp_sh + (tes_sh_o)
hp_sh_el = -hps_sh_inputs
#sto_sh_i = tes_sh_i
loa_sh = -demand_sh

#CO production
hp_co = hps_co
#sto_co_o = hp_co + (tes_co_o)
hp_co_el = -hps_co_inputs
#sto_co_i = tes_co_i
loa_co = -demand_co

#Electricity generation and consumption
pv = pv_rooftop 
grid = pv + grid
loa_el = -demand_el
hps_con = loa_el + hp_dhw_el + hp_sh_el + hp_co_el

#T trend
#T_ring_data = pd.DataFrame(T_ring_iter[-1][1:])#, index=COP_co_series.index)
#T_ring_data.set_index(cop_co_trend.index, inplace=True)


#%% Plots

day = '2015-01-01 00:00:00'
end = '2015-01-11 23:00:00'

#fig = plt.figure(figsize=(10,6))
fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(5,1, sharex = 'col', gridspec_kw = {'height_ratios':[1,1,1,1,1], 'wspace':0.1, 'hspace':0.1}, figsize=(12,16))

#ax1 = fig.add_subplot(111)
ax1.plot(loa_dhw[day:end].index,loa_dhw[day:end].values,'#000000', alpha=0.5, linestyle = '-', label ='DHW load')
ax1.plot(hp_dhw[day:end].index,hp_dhw[day:end].values,'#DC2A62', alpha=0.2)
#ax1.plot(sto_dhw_o[day:end].index,sto_dhw_o[day:end].values,'#E06E92', alpha=0.2)
#ax1.plot(sto_dhw_i[day:end].index,sto_dhw_i[day:end].values,'#E06E92', alpha=0.2)
#ax1.plot(hp_dhw_el[day:end].index,hp_dhw_el[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='Power consumption')
ax1.set_ylabel('Power (kW)',labelpad = 11)
#ax1.set_xlabel('Time (hours)')
#ax1.set_ylim(ymax = 28)
ax1.margins(x=0)
ax1.margins(y=0)
#ax1.set_xticks(np.arange(0,24,3))
#ax1.set_xticklabels(['0','3','6','9','12','15','18','21','24'])
ax1.fill_between(hp_dhw[day:end].index,0,hp_dhw[day:end].values,facecolor = '#DC2A62', alpha = 0.7, label = 'DHW Heat Pumps')
#ax1.fill_between(sto_dhw_o[day:end].index,hp_dhw[day:end].values,sto_dhw_o[day:end].values,facecolor = '#E06E92', alpha = 0.6, label = 'DHW Thermal storage')
#ax1.fill_between(sto_dhw_i[day:end].index,0,sto_dhw_i[day:end].values,facecolor = '#E06E92', alpha = 0.6)
lgd1 = ax1.legend(loc=0,  bbox_to_anchor=(1,1), fontsize='x-small')

ax11 = ax1.twinx()
#ax11.plot(tes_dhw_cap[day:end].index,tes_dhw_cap[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='DHW Storage')
ax11.set_ylabel('Stored Energy (kWh)',labelpad = 11)


#ax1 = fig.add_subplot(111)
ax2.plot(loa_sh[day:end].index,loa_sh[day:end].values,'#000000', alpha=0.5, linestyle = '-', label ='sh load')
ax2.plot(hp_sh[day:end].index,hp_sh[day:end].values,'#E23516', alpha=0.2)
#ax2.plot(sto_sh_o[day:end].index,sto_sh_o[day:end].values,'#E07F6E', alpha=0.2)
#ax2.plot(sto_sh_i[day:end].index,sto_sh_i[day:end].values,'#E07F6E', alpha=0.2)
#ax2.plot(hp_sh_el[day:end].index,hp_sh_el[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='Power consumption')
ax2.set_ylabel('Power (kW)',labelpad = 11)
#ax2.set_xlabel('Time (hours)')
#ax2.set_ylim(ymax = 28)
ax2.margins(x=0)
ax2.margins(y=0)
#ax2.set_xticks(np.arange(0,24,3))
#ax2.set_xticklabels(['0','3','6','9','12','15','18','21','24'])
ax2.fill_between(hp_sh[day:end].index,0,hp_sh[day:end].values,facecolor = '#E23516', alpha = 0.7, label = 'sh Heat Pumps')
#ax2.fill_between(sto_sh_o[day:end].index,hp_sh[day:end].values,sto_sh_o[day:end].values,facecolor = '#E07F6E', alpha = 0.6, label = 'sh Thermal storage')
#ax2.fill_between(sto_sh_i[day:end].index,0,sto_sh_i[day:end].values,facecolor = '#E07F6E', alpha = 0.6)
lgd2 = ax2.legend(loc=0,  bbox_to_anchor=(1,1), fontsize='x-small')

ax22 = ax2.twinx()
#ax22.plot(tes_sh_cap[day:end].index,tes_sh_cap[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='sh Storage')
ax22.set_ylabel('Stored Energy (kWh)',labelpad = 11)

##ax1 = fig.add_subplot(111)
ax3.plot(loa_co[day:end].index,loa_co[day:end].values,'#000000', alpha=0.5, linestyle = '-', label ='co load')
ax3.plot(hp_co[day:end].index,hp_co[day:end].values,'#2A63DC', alpha=0.2)
#ax3.plot(sto_co_o[day:end].index,sto_co_o[day:end].values,'#829FDF', alpha=0.2)
#ax3.plot(sto_co_i[day:end].index,sto_co_i[day:end].values,'#829FDF', alpha=0.2)
#ax3.plot(hp_co_el[day:end].index,hp_co_el[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='Power consumption')
ax3.set_ylabel('Power (kW)',labelpad = 11)
#ax3.set_xlabel('Time (hours)')
#ax3.set_ylim(ymax = 28)
ax3.margins(x=0)
ax3.margins(y=0)
#ax3.set_xticks(np.arange(0,24,3))
#ax3.set_xticklabels(['0','3','6','9','12','15','18','21','24'])
ax3.fill_between(hp_co[day:end].index,0,hp_co[day:end].values,facecolor = '#2A63DC', alpha = 0.7, label = 'co Heat Pumps')
#ax3.fill_between(sto_co_o[day:end].index,hp_co[day:end].values,sto_co_o[day:end].values,facecolor = '#829FDF', alpha = 0.6, label = 'co Thermal storage')
#ax3.fill_between(sto_co_i[day:end].index,0,sto_co_i[day:end].values,facecolor = '#829FDF', alpha = 0.6)
lgd2 = ax3.legend(loc=0,  bbox_to_anchor=(1,1), fontsize='x-small')

ax33 = ax3.twinx()
#ax33.plot(tes_co_cap[day:end].index,tes_co_cap[day:end].values,'#000000', alpha=0.5, linestyle = '--', label ='co Storage')
ax33.set_ylabel('Stored Energy (kWh)',labelpad = 11)


ax4.plot(hp_dhw_el[day:end].index,hp_dhw_el[day:end].values,'#000000', alpha=0.5, color = 'r', linestyle = '--', label ='HPs power consumption')
ax4.plot(loa_el[day:end].index,loa_el[day:end].values,'#000000', alpha=0.5, linestyle = '-', label ='Electricity demand')
ax4.plot(pv[day:end].index,pv[day:end].values,'#EBF125', alpha=0.2)
ax4.plot(grid[day:end].index,grid[day:end].values,'#8834C2', alpha=0.2)
ax4.set_ylabel('Power kW)',labelpad = 11)
#ax4.set_xlabel('Time (hours)')
#ax4.set_ylim(ymax = 28)
ax4.margins(x=0)
ax4.margins(y=0)
#ax4.set_xticks(np.arange(0,24,3))
#ax4.set_xticklabels(['0','3','6','9','12','15','18','21','24'])
ax4.fill_between(pv[day:end].index,0,pv[day:end].values,facecolor = '#EBF125', alpha = 0.6, label = 'Rooftop PV')
ax4.fill_between(grid[day:end].index,pv[day:end].values,grid[day:end].values,facecolor = '#8834C2', alpha = 0.6, label = 'Electricity grid')
lgd4 = ax4.legend(loc=0,  bbox_to_anchor=(1,1), fontsize='x-small')

#ax5.plot(hp_dhw_el[day:end].index,T_ring_data[day:end].values, 'r')

#%% csv export 
#storages = pd.DataFrame([tes_dhw_i.values, tes_dhw_o.values, tes_dhw_cap.values, tes_sh_i.values, tes_sh_o.values, tes_sh_cap.values, tes_co_i.values, tes_co_o.values, tes_co_cap.values]).T
#storages.columns=['tes_dhw_in','tes_dhw_out','tes_dhw_cap','tes_sh_in','tes_sh_out','tes_sh_cap','tes_co_in','tes_co_out','tes_co_cap']
#storages.set_index(hp_dhw_el.index, inplace=True)
#storages.to_csv('storages.csv')
#
#r_storages = pd.DataFrame([tes_dhw_i['buildings'].values, tes_dhw_o['buildings'].values, tes_dhw_cap['buildings'].values, tes_sh_i['buildings'].values, tes_sh_o['buildings'].values, tes_sh_cap['buildings'].values, tes_co_i['buildings'].values, tes_co_o['buildings'].values, tes_co_cap['buildings'].values, tes_dhw_i['commercial'].values, tes_dhw_o['commercial'].values, tes_dhw_cap['commercial'].values, tes_sh_i['commercial'].values, tes_sh_o['commercial'].values, tes_sh_cap['commercial'].values, tes_co_i['commercial'].values, tes_co_o['commercial'].values, tes_co_cap['commercial'].values]).T
#r_storages.columns=['res_tes_dhw_in','res_tes_dhw_out','res_tes_dhw_cap','res_tes_sh_in','res_tes_sh_out','res_tes_sh_cap','res_tes_co_in','res_tes_co_out','res_tes_co_cap', 'off_tes_dhw_in','off_tes_dhw_out','off_tes_dhw_cap','off_tes_sh_in','off_tes_sh_out','off_tes_sh_cap','off_tes_co_in','off_tes_co_out','off_tes_co_cap']
#r_storages.set_index(hp_dhw_el.index, inplace=True)
#r_storages.to_csv('storages_separated.csv')

hps_dhw.to_csv('hp_dhw_out.csv')
hps_dhw_inputs.to_csv('hp_dhw_in.csv')
#tes_dhw_o.to_csv('tes_dhw_out.csv')
#tes_dhw_i.to_csv('tes_dhw_in.csv')
#tes_dhw_cap.to_csv('tes_dhw_cap.csv')

hps_sh.to_csv('hp_sh_out.csv')
hps_sh_inputs.to_csv('hp_sh_in.csv')
#tes_sh_o.to_csv('tes_sh_out.csv')
#tes_sh_i.to_csv('tes_sh_in.csv')
#tes_sh_cap.to_csv('tes_sh_cap.csv')

hps_co.to_csv('hp_co_out.csv')
hps_co_inputs.to_csv('hp_co_in.csv')
#tes_co_o.to_csv('tes_co_out.csv')
#tes_co_i.to_csv('tes_co_in.csv')
#tes_co_cap.to_csv('tes_co_cap.csv')

grid.to_csv('grid_supply.csv')
pv_rooftop.to_csv('pv_supply.csv')