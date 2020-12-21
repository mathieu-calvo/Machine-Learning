import numpy as np 
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import ipywidgets as widgets
from IPython.display import display, HTML

        
def visualize_ml_dataset(df, label=None):
    """
    Explore machine learning dataset and give meaningful insights into content 
    of the dataset by providing interactive visualizations
    
    Args:
        - df (pd.DataFrame): tidy dataFrame with attributes (and label if supervised learning) 
        as columns, and observations as rows
        - label (str): column name that contain labels, default None for unsupervised learning
    """
    # see type of fields
    all_fields = df.columns.tolist()
    categorical_fields = df.loc[:, df.dtypes == object].columns.tolist()
    numerical_fields = [col for col in df.columns if col not in categorical_fields] 

    # params of analysis type
    type_analytics = {
        'features type': {'cols': 'all_fields'},
        '% missing values': {'cols': 'all_fields'},
        'missing values patterns': {'cols': 'all_fields'},
        'statistical summary': {'cols': 'numerical_fields'},
        'boxplots': {'cols': 'numerical_fields'},
        'parallel coordinates': {'cols': 'numerical_fields', 'module': px, 'func': 'parallel_coordinates'},
        'parallel categories': {'cols': 'categorical_fields', 'module': px, 'func': 'parallel_categories'},
        'scatter matrix': {'cols': 'numerical_fields', 'module': px, 'func': 'scatter_matrix'},
        'correl heatmap': {'cols': 'numerical_fields', 'module': go, 'func': 'Heatmap'},
    }

    # create dropdowns widgets
    x_widget = widgets.Dropdown(options=type_analytics.keys(), description="Visualization:")
    y_widget = widgets.SelectMultiple(options=eval(type_analytics[x_widget.value]['cols']), description="Features:")
    y_widget.value = eval(type_analytics[x_widget.value]['cols'])

    # Define a function that updates the content of y based on what we select for x
    def update(*args):
        selectable_columns = eval(type_analytics[x_widget.value]['cols'])
        y_widget.options = selectable_columns
        y_widget.value = selectable_columns
    x_widget.observe(update)
    
    # Some function you want executed
    def interact_with_visualizations(x, y):
        
        plot_type = x
        cols_selected = y
        
        # filter frame
        mdf = df[list(cols_selected)]

        # plot according to choice
        if 'module' in type_analytics[plot_type]:

            if type_analytics[plot_type]['module'] == px:
                if label not in list(cols_selected):
                    mdf = df[list(cols_selected) + [label]]
                method_to_call = getattr(px, type_analytics[plot_type]['func'])
                method_to_call(mdf, color=label).show()

            elif type_analytics[plot_type]['module'] == go:
                method_to_call = getattr(go, type_analytics[plot_type]['func'])
                cdf = mdf.corr().round(4)
                go.Figure(data=method_to_call(z=cdf.values, x=cdf.index, y=cdf.columns)).show()

        elif plot_type == '% missing values':
            pdf = (mdf.isnull().sum() / mdf.shape[0] * 100).sort_values(ascending=False).to_frame().round(1)
            pdf.columns = ['% missing']
            display(HTML(pdf.to_html()))

        elif plot_type == 'missing values patterns':
            fig = px.imshow(mdf.isnull());

        elif plot_type == 'statistical summary':
            display(HTML(mdf.describe().round(2).to_html()))

        elif plot_type == 'boxplots':
            fig = mdf.boxplot();

        elif plot_type == 'features type':
            ddf = mdf.dtypes.to_frame()
            ddf.columns = ['type']
            display(HTML(ddf.to_html()))

    return widgets.interactive(interact_with_visualizations, x=type_analytics.keys(), y=y_widget)