import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')


def get_kind(x: pd.Series, diff_limit: int = 8):
    x = x.astype('str')
    x = x.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    x.dropna(inplace=True)
    if x.nunique() > diff_limit:
        kind = 'numeric'
    else:
        kind = 'categorical'
    return kind

def check_data_y(X):
    """
    检查数据结构，数据预测变量为 0,1，并以“y”命名
    """
    if 'y' not in X.columns:
        logging.error('未检测到"y"变量，请将预测变量命名改为"y"')

class Feature_select(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 num_method: str = 'sys',
                 cate_method: str = 'sys',
                 diff_num: int = 10,
                 pos_label: str = 1,
                 show_df: bool = False):
        self.num_list = num_list
        self.cate_list = cate_list
        self.num_method = num_method
        self.cate_method = cate_method
        self.diff_num = diff_num
        self.pos_label = pos_label
        self.show_df = show_df
        self.select_list = []

    def fit(self, X, y=None):
        X = X.copy()
        from scipy import stats
        if self.num_list is None:
            self.num_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)
        if self.cate_list is None:
            self.cate_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'categorical':
                    self.cate_list.append(col)
        X['y'] = y
        yes = X[X['y'] == self.pos_label]
        yes.reset_index(drop=True, inplace=True)
        no = X[X['y'] != self.pos_label]
        no.reset_index(drop=True, inplace=True)
        del X['y']
        sys_cate_list, kf_list, kf_p_list = [], [], []
        sys_num_list, t_list, p_value_list, anova_f_list, anova_p_list = [], [], [], [], []
        if self.cate_method == 'sys' or self.show_df is True:
            for obj in self.cate_list:
                value_list = list(X[obj].unique())
                value_sum = 0
                for value in value_list:
                    support_yes = (yes[yes[obj] == value].shape[0] + 1) / (yes.shape[0] + 1)
                    support_no = (no[no[obj] == value].shape[0] + 1) / (no.shape[0] + 1)
                    confidence_yes = support_yes / (support_yes + support_no)
                    value_sum += abs(2 * confidence_yes - 1) * (X[X[obj] == value].shape[0] / X.shape[0])
                sys_cate_list.append(value_sum)
                if value_sum >= 0.1:
                    self.select_list.append(obj)

        if self.cate_method == 'kf' or self.show_df is True:
            for obj in self.cate_list:
                df_obj = pd.get_dummies(X[obj], prefix=obj)
                df_obj['result'] = y
                df_obj = df_obj.groupby('result').sum()
                obs = df_obj.values
                kf = stats.chi2_contingency(obs)
                '''
                chi2: The test statistic
                p: p-value
                dof: Degrees of freedom
                expected: The expected frequencies, based on the marginal sums of the table.
                '''
                chi2, p, dof, expect = kf
                kf_list.append(chi2)
                kf_p_list.append(p)

                if p < 0.05:
                    self.select_list.append(obj)

        if self.num_method == 'sys' or self.show_df is True:
            for num in self.num_list:
                mean_c1 = no[num].mean()
                std_c1 = no[num].std()
                mean_c2 = yes[num].mean()
                std_c2 = yes[num].std()
                value_sum = abs(mean_c1 - mean_c2) / (std_c1 + std_c2) * 2
                sys_num_list.append(value_sum)
                if value_sum >= 0.1:
                    self.select_list.append(num)

        if self.num_method == 't' or self.show_df is True:
            for num in self.num_list:
                t_t, t_p = stats.ttest_ind(yes[num], no[num], equal_var=False, nan_policy='omit')  # 'omit'忽略nan值执行计算
                t_list.append(t_t)
                p_value_list.append(t_p)
                if t_p < 0.05:
                    self.select_list.append(num)
                # print('attr=%s, t=%.5f, p=%.5f' % (num, t, p_value))

        if self.num_method == 'anova' or self.show_df is True:
            for num in self.num_list:
                anova_f, anova_p = stats.f_oneway(yes[num], no[num])
                anova_f_list.append(anova_f)
                anova_p_list.append(anova_p)
                # print('attr=%s, anova_f=%.5f, anova_p=%.5f' % (num, anova_f, anova_p))
                if anova_p < 0.05:
                    self.select_list.append(num)

        if self.show_df is True:
            dic1 = {'categorical': self.cate_list, 'importance_': sys_cate_list, 'Kf-Value': kf_list,
                    'Kf-P-Value': kf_p_list}
            df = pd.DataFrame(dic1, columns=['categorical', 'importance_', 'Kf-Value', 'Kf-P-Value'])
            df.sort_values(by='Kf-P-Value', inplace=True)
            print(df)
            dic2 = {'numeric': self.num_list, 'importance_': sys_num_list, 'T-Value': t_list, 'P-value': p_value_list,
                    'Anova-F-Value': anova_f_list, 'Anova-P-value': anova_p_list}
            df = pd.DataFrame(dic2,
                              columns=['numeric', 'importance_', 'T-Value', 'P-value', 'Anova-F-Value',
                                       'Anova-P-value'])
            df.sort_values(by='Anova-P-value', inplace=True)
            print(df)
        self.select_list = list(set(self.select_list))
        print('After select attr:', self.select_list)
        return self

    def transform(self, X):
        X = X.copy()
        logging.info('attr select success!')
        return X[self.select_list]
