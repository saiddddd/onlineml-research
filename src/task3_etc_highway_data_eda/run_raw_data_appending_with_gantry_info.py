#%%
import pandas as pd
import glob
#%%
df_gantry_info = pd.read_csv('./data/GantryID_info.csv')
df_gantry_info['里程'] = [x[-6:-1] for x in df_gantry_info['編號']]
aaa = df_gantry_info['國道別'].unique()

#%%
ic_order_dict = {}

for i in aaa:
    # print(i)
    # print(df_gantry_info['國道別'] == i)
    df_gantry_info_filter_highway_number = df_gantry_info[df_gantry_info['國道別'] == i]
    milestone = df_gantry_info_filter_highway_number['里程'].unique()
    
    for j in range(len(milestone)):
        ic_order_dict[str(milestone[j])] = j+1
    
print(ic_order_dict)

# %%
df_gantry_info['ICNUM'] = [ic_order_dict.get(x) for x in df_gantry_info['里程']]
df_gantry_info['ICOrder'] = df_gantry_info.apply(lambda x: x['ICNUM'] if x['方向'] == 'S' else -x['ICNUM'], axis=1)
df_gantry_info
#%%
for filename in glob.iglob('data/RawData/M05A/'+'**/*.csv', recursive=True):
    if '2021' in filename:
        df_traffic = pd.read_csv('./'+filename, names=["DateTime", "GantryID", "GantryTo", "VehicleType", "MeanSpeed", "Traffic"])
        df_traffic_dumpped = df_traffic[df_traffic['VehicleType']==31]
        df_merged = pd.merge(df_traffic_dumpped, df_gantry_info[['GantryID', '緯度', '經度','國道別', '方向', '起點交流道', '迄點交流道', '里程', 'ICNUM']], on='GantryID')
        df_merged.to_csv('./HighwayTrafficData_RawDataAppended_2021.csv', mode='a')
        
# %%

df_gantry_info.to_csv('GantryID_info_IC_order.csv')

# %%
