import pandas as pd

class loadBankDataset():
    def __init__(self):
        self.data = pd.read_csv("BankChurners.csv")

        self.data['BalanceSalaryRatio'] = self.data.Balance / self.data.EstimatedSalary
        self.data['hei'] = self.data['HasCrCard'] | self.data['Exited'] | self.data['IsActiveMember']
        self.data['ieh'] = self.data['HasCrCard'] & self.data['Exited'] & self.data['IsActiveMember']

        feature = ['CustomerId', 'Geography', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                   'IsActiveMember', 'EstimatedSalary', 'Exited']  # 9
        unwanted_cols = ['CustomerId','Balance','EstimatedSalary']
        # unwanted_cols = ['CustomerId']
        self.data = self.data.drop(unwanted_cols, axis=1)

        # Length of data
        self.len = len(self.data)

        # Create CreditLevel list for one hot encoding later
        self.credit_ls = self.data['CreditLevel'].unique()

        # Create Geography dict to map country to number
        unique_geo = self.data['Geography'].unique()
        self.geo_dict = dict()
        for idx, geo in enumerate(unique_geo):
            self.geo_dict[geo] = idx

        # Convert country to number
        self.data['Geography'] = self.data['Geography'].apply(lambda x: self.geo_dict[x])

        # Normalzie Balance and EstimatedSalary
        # self.data['Balance'] = (self.data['Balance'] - self.data['Balance'].min()) \
        #                        / (self.data['Balance'].max() - self.data['Balance'].min())
        self.data['BalanceSalaryRatio'] = (self.data['BalanceSalaryRatio'] - self.data['BalanceSalaryRatio'].min()) \
                               / (self.data['BalanceSalaryRatio'].max() - self.data['BalanceSalaryRatio'].min())

        # self.data['EstimatedSalary'] = (self.data['EstimatedSalary'] - self.data['EstimatedSalary'].min()) \
        #                                / (self.data['EstimatedSalary'].max() - self.data['EstimatedSalary'].min())



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Define CreditLevel as label
        label = self.data.iloc[idx, 6]  ##
        label = label - 1
        # Retreive attributes
        attribute = self.data.iloc[idx, [0,1,2,3,4,5,7,8,9]].to_numpy()  ##
        return attribute, label