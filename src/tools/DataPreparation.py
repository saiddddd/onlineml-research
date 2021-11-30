
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import os


class DataPreparation(ABC):
    """
    The Abstraction Factory interface declares a set of methods that return
    different abstract DataPreparer. A family of DataPreparer may have several variants.
    """

    @abstractmethod
    def __init__(self):
        self._df = pd.DataFrame()

    @abstractmethod
    def _data_prepare(self):
        pass

    def _data_object_encoder(self):
        #======================#
        # To do object encoder #
        #======================#
        for col in self._df.columns:
            if self._df[col].dtype == 'object':
                self._df[col] = self._df[col].fillna(self._df[col].mode())
                self._df[col] = LabelEncoder().fit_transform(self._df[col])
            else:
                self._df[col] = self._df[col].fillna(self._df[col].median())

    def check_target_rate(self):
        # self._data_prepare()
        target_rate = self._target[self._target == 1].count()/len(self._target)
        return target_rate

    def get_pd_df_data(self):
        # self._data_prepare()
        return self._df, self._target

    def get_splitted_train_test_pd_df_data(self,
                                           training_dataset_ratio=0.7,
                                           random_seed=42):
        """
        invoked concrete data prepare method for pre-processing data
        return splitted training dataset and testing dataset for doing
        ML pipeline
        """
        features_data, target_data = self.get_pd_df_data()

        from sklearn.model_selection import train_test_split
        train_features, test_features, train_target, test_target = train_test_split(
            features_data,
            target_data,
            test_size=1 - training_dataset_ratio,
            random_state=random_seed
        )
        return train_features, test_features, train_target, test_target


class CreditCardPreparation(DataPreparation):

    def __init__(self):
        '''
        Data initialization is doing aggregation in this case,
        due to credit dataframe ID is not unique, every credit data raw represent credit card monthly balance
        for each customer has multiple records,
        We have to do aggression first before doing aggregation.
        '''

        #==============#
        # Data Loading #
        #==============#
        df_app = pd.read_csv("/Users/pwang/BenWork/OnlineML/onlineml/data/credit_card_approvel/application_record.csv")
        df_credit = pd.read_csv("/Users/pwang/BenWork/OnlineML/onlineml/data/credit_card_approvel/credit_record.csv")

        # Convert status to numertic and group-max by status for each unique id.
        # This will be a proxy for whether an application will be approved, since there is no yes/no flag
        # X and C standing for Without loan for that Month, and paid off that Month respectively.
        df_credit['STATUS'] = df_credit['STATUS'].replace(['X'], 0)
        df_credit['STATUS'] = df_credit['STATUS'].replace(['C'], 0)
        df_credit['STATUS'] = df_credit['STATUS'].apply(pd.to_numeric)

        # ============================================================================================ #
        # Change of groupby policy,                                                                    #
        # Not only check STATUS alone, but also have to consider MONTHS_BALANCES history               #
        # Using this to check the old, and new customer, add dummy date time to do following PoC       #
        # ============================================================================================ #
        df_credit = df_credit.groupby('ID').agg({'STATUS': 'max', 'MONTHS_BALANCE': 'min'})[
            ['STATUS', 'MONTHS_BALANCE']
        ].reset_index()

        # join features and target (label)
        self._df = pd.merge(df_app, df_credit, left_on='ID', right_on='ID')

        self._data_prepare()

    def _data_prepare(self):
        """
        Operating on merged dataframe self._df
        re-formatted it to pre-dataframe ready for applied on Model Training
        split feature and target
        """

        # defining the target, risk,
        # Check out STATUS to decide weather this customer is High Risk
        # If STATUS < 1 return 0 (no risk); else return 1 (with risk)
        self._df['high_risk'] = np.where(self._df['STATUS'] < 1, 0, 1)
        # convert days old to years
        self._df['age_years'] = round(self._df['DAYS_BIRTH'] / -365, 0).astype(int)
        self._df['years_employed'] = round(self._df['DAYS_EMPLOYED'] / -365, 0).astype(int)

        # encoding category column
        self._df = pd.get_dummies(
            self._df, columns=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'],
            prefix=["gender", "own_car", "own_property", "income_type", "edication", "family_status", "housing_type", "occupation_type"]
        )

        #drop columns not needed
        self._df.drop(['ID'], axis=1, inplace=True)
        self._df.drop(['STATUS'], axis=1, inplace=True)
        self._df.drop(['DAYS_BIRTH'], axis=1, inplace=True)
        self._df.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)
        self._df.drop(['own_car_N'], axis=1, inplace=True)
        self._df.drop(['own_property_N'], axis=1, inplace=True)

        self._df = self._df.sort_values(by=['MONTHS_BALANCE'], ascending=False)

        # split target from features
        self._target = self._df.pop('high_risk')


class AirlineDataPreparation(DataPreparation):

    def __init__(self):

        #==============#
        # Data loading #
        #==============#
        self._df = pd.read_csv("/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv")
        self._data_prepare()

    def _data_prepare(self):
        self._data_object_encoder()

        self._target = self._df.pop('satisfaction')


class ArbitraryDataPreparation(DataPreparation):
    def __init__(self, input_path: str = '', label_name: str = '', feature_to_drop: list = []):
        try:
            self._df = pd.read_csv(input_path)
            self._data_prepare(label_name)
        except:
            raise FileNotFoundError


        print(self._df.columns)

        if len(feature_to_drop) > 0:
            for i in feature_to_drop:
                if isinstance(i, str):
                    try:
                        aaa = self._df.pop(i)
                        print('preparation of dataframe, drop feature {} successfully'.format(i))
                    except :
                        print('feature going to drop {} not found'.format(i))

        print(self._df.columns)


    def _data_prepare(self, label_name):
        self._data_object_encoder()
        try:
            self._target = self._df.pop(label_name)
        except:
            print('dataframe can not found the target label {}'.format(label_name))
            raise RuntimeError

if __name__ == '__main__':
    from tools.DataVisualization import HistogramCompare, TrendPlot
    features_data, target_data = CreditCardPreparation().get_pddataframe_data_by_month_balance_order()

    split_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    training_target_rate = []
    testing_target_rate = []

    from sklearn.model_selection import train_test_split
    train_features, test_features, train_target, test_target = train_test_split(
        features_data,
        target_data,
        test_size=0.5,
        # random_state=42
        shuffle=False,
        stratify=None
    )

    TrendPlot().plot_trend(np.array(train_features['MONTHS_BALANCE']), y_label='month recode')
    TrendPlot().plot_trend(np.array(test_features['MONTHS_BALANCE']), y_label='month recode')

    print("Training dataset target rate: {} %".format(train_target[train_target == 1].count() / len(train_target) * 100))
    print("Testing dataset target rate: {} %".format(test_target[test_target == 1].count() / len(test_target) * 100))

    # print(features_data.count())

    # data_visualizer = HistogramCompare(np.array(train_features['age_years']), np.array(test_features['age_years']))
    # data_visualizer.draw_nominal_comparing_dist(data_name='Age Years', density=True)



def test_get_pddataframe_data_by_month_balance_order():
    features, target = CreditCardPreparation().get_pd_df_data()
    # f = open('./df_feature_Describe.txt', 'a')
    # f.write(features.describe())
    # f.close()
    features.info()
    print(features['AMT_INCOME_TOTAL'].describe())
    # np.savetxt(r'df_feature_Describe.txt', features.describe(), fmt='%d')
    print(target.describe())
    assert True