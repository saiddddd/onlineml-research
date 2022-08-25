 # %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


#======================#
# input data after eda #
#======================#

ana_df = pd.read_csv('../../data/highway/highway_traffic_eda_data_ready_for_ml_2021_01.csv')

#========================================#
# Aggregting traffic data by time and ic #
#========================================#
aaa = ana_df.groupby(
    [ana_df['DayOfWeek'], ana_df.Time.dt.time, ana_df['GantryID'], ana_df['緯度'], ana_df['經度'], ana_df['起點交流道'], ana_df['迄點交流道']]
    ).agg(
        {
            'Traffic': np.mean,
            'MeanSpeed': np.mean,
        }
        ).reset_index()
aaa['Time'] = pd.to_datetime(aaa['Time'], format='%H:%M:%S')

#==============#
# Ploting data #
#==============#
fig, ax = plt.subplots(2, 2)
fig.suptitle('2021/01/01 ~ 2021/05/01')
fig.autofmt_xdate()
fig.set_figheight(12)
fig.set_figwidth(20)

dtFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(dtFmt)

ax[0, 0].grid(True)
ax[0, 0].set_title('IC : Taipei')
ax[0, 0].hist(aaa[aaa.GantryID == '01F0155S']['Time'], 288, weights=aaa[aaa.GantryID == '01F0155S']['Traffic'], label='Traffics')
ax[0, 0].set_ylabel('Traffics by 5 mins')
ax2 = ax[0, 0].twinx()
ax2.set_ylabel('MeanSpeed', color='r')
ax2.plot(aaa[(aaa.GantryID == '01F0155S') & (aaa.DayOfWeek == 1)]['Time'], aaa[(aaa.GantryID == '01F0155S') & (aaa.DayOfWeek == 1)]['MeanSpeed'], color='r', label='MeanSpeed')
ax2.tick_params(axis='y', labelcolor='r')

trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
ax2.fill_between(aaa[(aaa.GantryID == '01F0155S') & (aaa.DayOfWeek == 1)]['Time'], 0, 100, where=aaa[(aaa.GantryID == '01F0155S') & (aaa.DayOfWeek == 1)]['MeanSpeed']<60, facecolor='r', alpha=0.4, transform=trans )

plt.gca().xaxis.set_major_formatter(dtFmt)
ax2.legend()

ax[1, 0].grid(True)
ax[1, 0].set_title('IC : Hsinchu')
ax[1, 0].hist(aaa[aaa.GantryID == '01F0928S']['Time'], 288, weights=aaa[aaa.GantryID == '01F0928S']['Traffic'], label='Traffics')
ax[1, 0].set_ylabel('Traffics by 5 mins')
ax2 = ax[1, 0].twinx()
ax2.set_ylabel('MeanSpeed', color='r')
ax2.plot(aaa[(aaa.GantryID == '01F0928S') & (aaa.DayOfWeek == 1)]['Time'], aaa[(aaa.GantryID == '01F0928S') & (aaa.DayOfWeek == 1)]['MeanSpeed'], color='r', label='MeanSpeed')
ax2.tick_params(axis='y', labelcolor='r')
plt.gca().xaxis.set_major_formatter(dtFmt)
ax2.legend()

ax[0, 1].grid(True)
ax[0, 1].set_title('IC : Taichung')
ax[0, 1].hist(aaa[aaa.GantryID == '01F1699S']['Time'], 288, weights=aaa[aaa.GantryID == '01F1699S']['Traffic'], label='Traffics')
ax[0, 1].set_ylabel('Traffics by 5 mins')
ax2 = ax[0, 1].twinx()
ax2.set_ylabel('MeanSpeed', color='r')
ax2.plot(aaa[(aaa.GantryID == '01F1699S') & (aaa.DayOfWeek == 1)]['Time'], aaa[(aaa.GantryID == '01F1699S') & (aaa.DayOfWeek == 1)]['MeanSpeed'], color='r', label='MeanSpeed')
ax2.tick_params(axis='y', labelcolor='r')
plt.gca().xaxis.set_major_formatter(dtFmt)
ax2.legend()

ax[1, 1].grid(True)
ax[1, 1].set_title('IC : Yunlin')
ax[1, 1].hist(aaa[aaa.GantryID == '01F2394S']['Time'], 288, weights=aaa[aaa.GantryID == '01F2394S']['Traffic'], label='Traffics')
ax[1, 1].set_ylabel('Traffics by 5 mins')
ax2 = ax[1, 1].twinx()
ax2.set_ylabel('MeanSpeed', color='r')
ax2.plot(aaa[(aaa.GantryID == '01F2394S') & (aaa.DayOfWeek == 1)]['Time'], aaa[(aaa.GantryID == '01F2394S') & (aaa.DayOfWeek == 1)]['MeanSpeed'], color='r', label='MeanSpeed')
ax2.tick_params(axis='y', labelcolor='r')
plt.gca().xaxis.set_major_formatter(dtFmt)
ax2.legend()
plt.show()
# %%
