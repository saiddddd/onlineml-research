#%% 
from curses import raw
import pandas as pd
import numpy as np

#%%
raw_stock_df = pd.read_csv("../../../data/stock_index_predict/append_0050_top30_trading_data.csv")

raw_gold_price_df = pd.read_csv("../../../data/stock_index_predict/gold_price_data.csv")
raw_oil_price_df = pd.read_csv("../../../data/stock_index_predict/oil_price_data.csv")
raw_usd_ntd_x_rate_df = pd.read_csv("../../../data/stock_index_predict/USD_NTD_X_data.csv")

raw_stock_df = raw_stock_df.merge(raw_gold_price_df[["Date", "Gold_Close"]], on='Date')
raw_stock_df = raw_stock_df.merge(raw_oil_price_df[["Date", "Oil_Close"]], on='Date')
raw_stock_df = raw_stock_df.merge(raw_usd_ntd_x_rate_df[["Date", "USD_NTD_X_Close"]], on='Date')

#%%

raw_stock_df

#%%

raw_stock_df.dropna(inplace=True)

# calculating moving average
MA_days = [10, 20, 30, 90, 180]
for ma in MA_days:
    ma_price_str = "MA{}Price".format(ma)
    ma_vol_str = "MA{}Volume".format(ma)
    
    raw_stock_df[ma_price_str] = raw_stock_df['Close'].rolling(ma).mean()
    raw_stock_df['DiffTo{}'.format(ma_price_str)] = (raw_stock_df['Close'] - raw_stock_df[ma_price_str]) / raw_stock_df['Close']
    
    raw_stock_df[ma_vol_str] = raw_stock_df['Volume'].rolling(ma).mean()
    raw_stock_df['DiffTo{}'.format(ma_vol_str)] = (raw_stock_df['Volume'] - raw_stock_df[ma_vol_str]) / raw_stock_df['Volume']

    #clean the absolute value of MA, leaving the informaction of relative current position and MA just enought for tree-based model analysis
    raw_stock_df.drop([ma_price_str], axis=1, inplace=True)
    raw_stock_df.drop([ma_vol_str], axis=1, inplace=True)

raw_stock_df['InterDayDiffRatio'] = (raw_stock_df['High'] - raw_stock_df['Low']).abs() / raw_stock_df['Close']
raw_stock_df['OpenToCloseRatio'] = (raw_stock_df['Open'] - raw_stock_df['Close']) / raw_stock_df['Open']
raw_stock_df['OpenToHighRatio'] = (raw_stock_df['Open'] - raw_stock_df['High']).abs() / raw_stock_df['Open']
raw_stock_df['OpenToLowRatio'] = (raw_stock_df['Open'] - raw_stock_df['Low']).abs() / raw_stock_df['Open']
raw_stock_df['HighToCloseRatio'] = (raw_stock_df['Close'] - raw_stock_df['High']).abs() / raw_stock_df['Close']
raw_stock_df['LowToCloseRatio'] = (raw_stock_df['Close'] - raw_stock_df['Low']).abs() / raw_stock_df['Close']

raw_stock_df['InterDayDiffRatio_1D_ago'] = raw_stock_df['InterDayDiffRatio'].shift(1)
raw_stock_df['OpenToCloseRatio_1D_ago'] = raw_stock_df['OpenToCloseRatio'].shift(1)
raw_stock_df['OpenToHighRatio_1D_ago'] = raw_stock_df['OpenToHighRatio'].shift(1)
raw_stock_df['OpenToLowRatio_1D_ago'] = raw_stock_df['OpenToLowRatio'].shift(1)
raw_stock_df['HighToCloseRatio_1D_age'] = raw_stock_df['HighToCloseRatio'].shift(1)
raw_stock_df['LowToCloseRatio_1D_ago'] = raw_stock_df['LowToCloseRatio'].shift(1)

raw_stock_df['InterDayDiffRatio_2D_ago'] = raw_stock_df['InterDayDiffRatio'].shift(2)
raw_stock_df['OpenToCloseRatio_2D_ago'] = raw_stock_df['OpenToCloseRatio'].shift(2)
raw_stock_df['OpenToHighRatio_2D_ago'] = raw_stock_df['OpenToHighRatio'].shift(2)
raw_stock_df['OpenToLowRatio_2D_ago'] = raw_stock_df['OpenToLowRatio'].shift(2)
raw_stock_df['HighToCloseRatio_2D_age'] = raw_stock_df['HighToCloseRatio'].shift(2)
raw_stock_df['LowToCloseRatio_2D_ago'] = raw_stock_df['LowToCloseRatio'].shift(2)

# cauculating daily return

raw_stock_df['Gold_DailyReturn'] = raw_stock_df['Gold_Close'].pct_change()
raw_stock_df['Oil_DailyReturn'] = raw_stock_df['Oil_Close'].pct_change()

raw_stock_df['DailyReturn'] = raw_stock_df['Close'].pct_change()

raw_stock_df['LABEL'] = ( raw_stock_df['DailyReturn'] > 0).astype(int)


# %%

raw_stock_df.drop(["Open"], axis=1, inplace=True)
raw_stock_df.drop(["High"], axis=1, inplace=True)
raw_stock_df.drop(["Low"], axis=1, inplace=True)
raw_stock_df.drop(["Close"], axis=1, inplace=True)
raw_stock_df.drop(["Volume"], axis=1, inplace=True)



#%%
raw_stock_df
#%%

final_to_save = raw_stock_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
#%%
# final_to_save.drop(['DailyReturn'], inplace=True)
# %%
raw_stock_df.to_csv('./eda_TW50_top30_append.csv', index=False)

# %%
