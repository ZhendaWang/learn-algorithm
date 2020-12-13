# 导入包
import numpy as np
import pandas as pd

import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

# 定义类
class EDAnalysis:
    def __init__(self,
                 data = None,
                 id_col: str = None,
                 target: str = None,
                 cate_list: list = None,
                 num_list: list = None,
                 ):
        self.data = data
        self.id_col = id_col
        self.target = target
        self.num_list = num_list
        self.cate_list = cate_list

    def draw_bar(self, colname: str):
        bar_num = self.data[colname].value_counts()
        # 条形图
        trace0 = go.Bar(x=bar_num.index.tolist(),
                        y=bar_num.values.tolist(),
                        text=bar_num.values.tolist(),
                        textposition='auto',
                        marker=dict(color=["blue", "red", "green", "indianred", "darkgrey"], opacity=0.5)
                        )
        data = [trace0]
        layout = go.Layout(title=f'Distribution_num of {colname}', bargap=0.4, height=600,
                           xaxis={'title': colname})
        fig = go.Figure(data=data, layout=layout)

        return fig

    def draw_pie(self, colname: str):
        pie_num = self.data[colname].value_counts()
        # 饼图
        trace1 = go.Pie(labels=pie_num.index.tolist(),
                        values=pie_num.values.tolist(),
                        hole=.5,
                        marker=dict(line=dict(color='white', width=1.3))
                        )
        data = [trace1]
        layout = go.Layout(title=f'Distribution_ratio of {colname}', height=600)
        fig = go.Figure(data=data, layout=layout)
        return fig

    def draw_bar_stack_cat(self, colname: str):
        # 交叉表
        cross_table = round(pd.crosstab(self.data[colname], self.data[self.target], normalize='index') * 100, 2)

        # 索引
        index_cols = cross_table.columns.tolist()

        # 轨迹列表
        data = []
        for i in index_cols:
            trace = go.Bar(x=cross_table[i].values.tolist(),
                           y=cross_table.index.tolist(),
                           name=str(i),
                           orientation='h',
                           marker={'opacity': 0.8}
                           )
            data.append(trace)

            # 布局
        layout = go.Layout(title=f'Relationship Between {cross_table.index.name} and {cross_table.columns.name}',
                           bargap=0.4,
                           barmode='stack',
                           height=600,
                           xaxis={'title': '百分比'},
                           yaxis={'title': colname}
                           )

        # 画布
        fig = go.Figure(data=data, layout=layout)
        return fig

    def draw_Histogram(self, colname: str):
        trace = go.Histogram(x=self.data[colname], histnorm='probability', opacity=0.8)

        data = [trace]
        layout = go.Layout(title=f'Histogram of {colname}', height=600,
                           xaxis={'title': colname})

        fig = go.Figure(data=data, layout=layout)
        return fig

    def draw_bar_stack_num(self, colname: str, bins_num:int = 25):
        # 交叉表
        x_data = pd.cut(self.data[colname], bins=bins_num)
        cross_table = round(pd.crosstab(x_data, self.data[self.target], normalize='index') * 100, 2)

        # 索引
        index_cols = cross_table.columns.tolist()

        # 轨迹列表
        data = []
        for i in index_cols:
            trace = go.Bar(x=cross_table.index.astype('str').tolist(),
                           y=cross_table[i].values.tolist(),
                           name=str(i),
                           orientation='v',
                           marker={'opacity': 0.8},
                           )
            data.append(trace)

            # 布局
        layout = go.Layout(title=f'Relationship Between {cross_table.index.name} and {cross_table.columns.name}',
                           bargap=0,
                           barmode='stack',
                           height=600,
                           xaxis={'title': colname},
                           yaxis={'title': '百分比'}
                           )
        # 画布
        fig = go.Figure(data=data, layout=layout)
        return fig

    def draw_scatter_matrix(self):
        # 目标
        index_vals = self.data[self.target].astype('category').cat.codes

        dimension_list = []

        for i in self.num_list:
            dimension_list.append(dict(label=i, values=self.data[i]))

        trace = go.Splom(dimensions=dimension_list,
                         text=self.data[self.target],
                         marker=dict(color=index_vals,
                                     showscale=False,  # colors encode categorical variables
                                     line_color='white', line_width=0.5)
                         )
        data = [trace]
        layout = go.Layout(title='Scatterplot Matrix Between numeric Attributes', height=600)
        fig = go.Figure(data=data, layout=layout)
        return fig

