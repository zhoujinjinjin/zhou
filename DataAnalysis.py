import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv('./BankChurners.csv')
    print(data.shape)

    print(data.isnull().sum())
    # no missing value

    # print(data.nunique())
    data = data.drop('CustomerId', axis=1)

    print(data.dtypes)

    # for i in range(10):
    #     x = (data.Exited[data['CreditLevel']==i].count())/9000
    #     x_p = '{:.2%}'.format(x)
    #     print(i,':\t',x_p)

    unique_geo = data['Geography'].unique()
    geo_dict = dict()
    for idx, geo in enumerate(unique_geo):
        geo_dict[geo] = idx

    data['Geography'] = data['Geography'].apply(lambda x: geo_dict[x])
    # data['Balance'] = (data['Balance'] - data['Balance'].min()) \
    #                        / (data['Balance'].max() - data['Balance'].min())
    #
    #
    # data['EstimatedSalary'] = (data['EstimatedSalary'] - data['EstimatedSalary'].min()) \
    #                                / (data['EstimatedSalary'].max() - data['EstimatedSalary'].min())


    #  Discrete value
    # fig, axarr = plt.subplots(2,2,figsize=(30, 12))
    #
    # sns.countplot(x='CreditLevel', hue='Geography', data=data, ax=axarr[0][0])
    # sns.countplot(x='CreditLevel', hue='HasCrCard', data=data, ax=axarr[0][1])
    # sns.countplot(x='CreditLevel', hue='IsActiveMember', data=data, ax=axarr[1][0])
    # sns.countplot(x='CreditLevel', hue='Exited', data=data, ax=axarr[1][1])

    #  Continuous value
    # plt.legend([], [], frameon=False)

    # fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
    #
    # sns.boxplot(y='Tenure', x='CreditLevel', hue='CreditLevel', data=data, ax=axarr[0][0])
    # sns.boxplot(y='Balance', x='CreditLevel', hue='CreditLevel', data=data, ax=axarr[0][1])
    # sns.boxplot(y='NumOfProducts', x='CreditLevel', hue='CreditLevel', data=data, ax=axarr[1][0])
    # sns.boxplot(y='EstimatedSalary', x='CreditLevel', hue='CreditLevel', data=data, ax=axarr[1][1])


    data['BalanceSalaryRatio'] = data.Balance/data.EstimatedSalary
    sns.boxplot(y='BalanceSalaryRatio', x='CreditLevel', hue='CreditLevel', data=data)

    # sns.scatterplot(x='Balance', y='EstimatedSalary', data=data)

    plt.show()

