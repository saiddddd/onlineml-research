#%% 
from curses import raw
import pandas as pd
import numpy as np

#%%
raw_dow30_df = pd.read_csv("../../../data/stock_index_predict/TWII.csv")
raw_dow30_df.dropna(inplace=True)

# calculating moving average
MA_days = [10, 20, 30]
for ma in MA_days:
    ma_price_str = "MA{}Price".format(ma)
    ma_vol_str = "MA{}Volume".format(ma)
    
    raw_dow30_df[ma_price_str] = raw_dow30_df['Close'].rolling(ma).mean()
    raw_dow30_df['DiffTo{}'.format(ma_price_str)] = (raw_dow30_df['Close'] - raw_dow30_df[ma_price_str]) / raw_dow30_df['Close']
    
    raw_dow30_df[ma_vol_str] = raw_dow30_df['Volume'].rolling(ma).mean()
    raw_dow30_df['DiffTo{}'.format(ma_vol_str)] = (raw_dow30_df['Volume'] - raw_dow30_df[ma_vol_str]) / raw_dow30_df['Volume']

    #clean the absolute value of MA, leaving the informaction of relative current position and MA just enought for tree-based model analysis
    raw_dow30_df.drop([ma_price_str], axis=1, inplace=True)
    raw_dow30_df.drop([ma_vol_str], axis=1, inplace=True)

raw_dow30_df['InterDayDiffRatio'] = (raw_dow30_df['High'] - raw_dow30_df['Low']).abs() / raw_dow30_df['Close']
raw_dow30_df['OpenToCloseRatio'] = (raw_dow30_df['Open'] - raw_dow30_df['Close']) / raw_dow30_df['Open']
raw_dow30_df['OpenToHighRatio'] = (raw_dow30_df['Open'] - raw_dow30_df['High']).abs() / raw_dow30_df['Open']
raw_dow30_df['OpenToLowRatio'] = (raw_dow30_df['Open'] - raw_dow30_df['Low']).abs() / raw_dow30_df['Open']
raw_dow30_df['HighToCloseRatio'] = (raw_dow30_df['Close'] - raw_dow30_df['High']).abs() / raw_dow30_df['Close']
raw_dow30_df['LowToCloseRatio'] = (raw_dow30_df['Close'] - raw_dow30_df['Low']).abs() / raw_dow30_df['Close']

# cauculating daily return
raw_dow30_df['DailyReturn'] = raw_dow30_df['Close'].pct_change()

raw_dow30_df['LABEL'] = ( raw_dow30_df['DailyReturn'] > 0).astype(int)


# %%

raw_dow30_df.drop(["Open"], axis=1, inplace=True)
raw_dow30_df.drop(["High"], axis=1, inplace=True)
raw_dow30_df.drop(["Low"], axis=1, inplace=True)
raw_dow30_df.drop(["Close"], axis=1, inplace=True)
raw_dow30_df.drop(["Volume"], axis=1, inplace=True)



#%%
raw_dow30_df
#%%

final_to_save = raw_dow30_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
#%%
final_to_save.drop(['DailyReturn'], inplace=True)
# %%
raw_dow30_df.to_csv('./eda_TWII.csv', index=False)

# %%
