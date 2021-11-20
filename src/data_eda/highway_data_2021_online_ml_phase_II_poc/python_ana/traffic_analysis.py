#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates

# %%
'''Reading Data from ETC dataset'''
# df_etc_traffic = pd.read_csv('../data/highway_traffic_speed.csv', parse_dates=['DateTime'])
# adding milestone and ic order information ...
df_etc_traffic = pd.read_csv('../data/highway_traffic_speed_ic_order.csv', parse_dates=['DateTime'])
# %%
df_etc_traffic
#%%

# %%
''' Selecting the region and direction we are interesting in'''
df1 = df_etc_traffic[
    (df_etc_traffic['DateTime']<'2021/05/01 00:00')
    # (df_etc_traffic['國道別'] == '國道一號') &
    # (df_etc_traffic['方向'] == 'N')
    ]

# %%
df2 = df_etc_traffic[
    (df_etc_traffic['DateTime']>'2021/05/10 01:00') &
    (df_etc_traffic['DateTime']<'2021/07/01 01:00')
    ]

# %%
def dataframe_transformer(df):

    #==============#
    # Type Casting #
    #==============#
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Traffic'] = df['Traffic'].astype(int)
    df['MeanSpeed'] = df['MeanSpeed'].astype(int)
    df['ICNUM'] = df['ICNUM'].astype(int)
    
    #===========================#
    # Add arrtibutes and labels #
    #===========================#
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    
    df['ICOrder'] = df.apply(lambda x: x['ICNUM'] if x['方向'] == 'S' else -x['ICNUM'], axis=1)
    df = df.sort_values(['DateTime', '國道別', '方向', 'ICOrder'], ascending=[True, True, True, True])
    df.reset_index()
    
    #======================================#
    # more traffic information from others #
    #======================================#
    
    # Get Upstreaming traffic
    # df['UpstreamingICTraffic'] = [ df[df['ICNUM'] == (df['ICNUM'])-1]]
    

    #==============================#
    # Add label -> is Traffic jam? #
    #==============================#
    df['TrafficJam'] = (df['MeanSpeed']<60) & (df['Traffic'] > 10)
    
    return df

df1 = dataframe_transformer(df1)
# %%
df1
#%%
#====================================================#
# Adding upstream and downstream data as new columns #
#====================================================#

#-----------------------------------------------#
# Special information, previoud IC or next IC ? #
#-----------------------------------------------#
df1['Upstream1Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(1)
df1['Upstream2Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(2)
df1['Upstream3Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(3)

df1['Upstream1MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(1)
df1['Upstream2MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(2)
df1['Upstream3MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(3)

df1['Downstream1Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(-1)
df1['Downstream2Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(-2)
df1['Downstream3Traffic'] = df1.groupby(['Date', '國道別', '方向'])['Traffic'].shift(-3)

df1['Downstream1MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(-1)
df1['Downstream2MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(-2)
df1['Downstream3MeanSpeed'] = df1.groupby(['Date', '國道別', '方向'])['MeanSpeed'].shift(-3)

#------------------------------------------#
# Time information, 10min ago, 30 min ago? #
#------------------------------------------#

df1['Traffic10MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['Traffic'].shift(2)
df1['Traffic30MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['Traffic'].shift(6)
df1['Traffic60MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['Traffic'].shift(12)

df1['MeanSpeed10MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['MeanSpeed'].shift(2)
df1['MeanSpeed30MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['MeanSpeed'].shift(6)
df1['MeanSpeed60MinAgo'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['MeanSpeed'].shift(12)

#---------------------------------#
# Is Traffic Jam in 30 min later? # 
#---------------------------------#
# %%
df1['TrafficJam30MinLater'] = df1.groupby(['國道別', 'GantryID', 'GantryTo'])['TrafficJam'].shift(-6)
# %%
df1['TrafficJam30MinLater'] = df1['TrafficJam30MinLater'].fillna(False)
# %%
df1['TrafficJam30MinLater'] = df1['TrafficJam30MinLater'].astype(int)
# %%
df1

# %%
df1.to_csv('test.csv')

# df1['Upstream2MeanTraffic'] = df1['Traffic'].rolling(2).mean().shift()
# %%
# df1.columns
df1_save = df1[['國道別', '方向', 'ICNUM', 'DayOfWeek', 'Hour', 'Minute', 
               'MeanSpeed', 'Traffic', 
               'Upstream1Traffic', 'Upstream2Traffic', 'Upstream3Traffic', 
               'Upstream1MeanSpeed', 'Upstream2MeanSpeed', 'Upstream3MeanSpeed',
               'Downstream1Traffic', 'Downstream2Traffic', 'Downstream3Traffic',
               'Downstream1MeanSpeed', 'Downstream2MeanSpeed', 'Downstream3MeanSpeed',
               'Traffic10MinAgo', 'Traffic30MinAgo', 'Traffic60MinAgo',
               'MeanSpeed10MinAgo', 'MeanSpeed30MinAgo', 'MeanSpeed60MinAgo',
               'TrafficJam', 'TrafficJam30MinLater'
               ]]
df1_save.to_csv('highway_traffic_analized_data_ready_for_ML.csv')
#%% 
# df2['DateTime'] = pd.to_datetime(df2['DateTime'])
# df2 = df2.sort_values(by='DateTime')
# df2['DayOfWeek'] = df2['DateTime'].dt.dayofweek
# df2['Traffic'] = df2['Traffic'].astype(int)
# df2['MeanSpeed'] = df2['MeanSpeed'].astype(int)
# df2['TrafficJam'] = (df2['MeanSpeed']<60) & (df2['Traffic'] > 10)

df2 = dataframe_transformer(df2)

# %%
aaa = df1.groupby(
    [df1['DayOfWeek'], df1.Time.dt.time, df1['GantryID'], df1['緯度'], df1['經度'], df1['起點交流道'], df1['迄點交流道']]
    ).agg(
        {
            'Traffic': np.mean,
            'MeanSpeed': np.mean,
        }
        ).reset_index()
aaa['Time'] = pd.to_datetime(aaa['Time'], format='%H:%M:%S')

# %%
bbb = df2.groupby(
    [df2['DayOfWeek'], df2.Time.dt.time, df2['GantryID'], df2['緯度'], df2['經度'], df2['起點交流道'], df2['迄點交流道']]
    ).agg(
        {
            'Traffic': np.mean,
            'MeanSpeed': np.mean,
        }
        ).reset_index()
bbb['Time'] = pd.to_datetime(bbb['Time'], format='%H:%M:%S')
# %%
import matplotlib.transforms as mtransforms

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

def plot_traffic_jam_analysis(plt, df, ax, ic_name='unknown IC'):
    ax.grid(True)
    ax.set_title('IC : {}'.format(ic_name))
    ax.hist(df['Time'], 288, weights=df['Traffic'], label='Traffics')
    ax.set_ylabel('Traffics by 5 mins')
    ax2 = ax.twinx()
    ax2.set_ylabel('MeanSpeed', color='r')
    ax2.set_ylim([10, 130])
    ax2.plot(df['Time'], df['MeanSpeed'], color='r', label='MeanSpeed')
    ax2.tick_params(axis='y', labelcolor='r')

    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.fill_between(df['Time'], 0, 1, where=df['MeanSpeed']<60, facecolor='r', alpha=0.4, transform=trans )

    plt.gca().xaxis.set_major_formatter(dtFmt)
    ax2.legend()
    return ax

week_dic={
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}

for i in range(0, 7):
    DOW = i
    

    fig, ax = plt.subplots(2, 2)
    fig.autofmt_xdate()
    fig.set_figheight(12)
    fig.set_figwidth(20)

    dtFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(dtFmt)
    fig.suptitle('2021/05/10 ~ 2021/07/01, Day of week:{}'.format(week_dic.get(DOW)))

    df_taipei_mon = bbb[(bbb.GantryID == '01F0155S') & (bbb.DayOfWeek == DOW)]
    df_hsinchu_mon = bbb[(bbb.GantryID == '01F0928S') & (bbb.DayOfWeek == DOW)]
    df_taichung_mon = bbb[(bbb.GantryID == '01F1699S') & (bbb.DayOfWeek == DOW)]
    df_yunlin_mon = bbb[(bbb.GantryID == '01F2394S') & (bbb.DayOfWeek == DOW)]

    ax[0, 0] = plot_traffic_jam_analysis(plt, df_taipei_mon, ax[0, 0], ic_name='Taipei')
    ax[0, 1] = plot_traffic_jam_analysis(plt, df_hsinchu_mon, ax[0, 1], ic_name='Hsunchu')
    ax[1, 0] = plot_traffic_jam_analysis(plt, df_taichung_mon, ax[1, 0], ic_name='Taichung')
    ax[1, 1] = plot_traffic_jam_analysis(plt, df_yunlin_mon, ax[1, 1], ic_name='Yunlin')
    plt.show()
# %%


