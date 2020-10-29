####
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Analyzer:
    def __init__(self):
        
        """Class for extracting specific properties of the data.
     
        Attribute: None
        Methods:
            check_null
            plot_feature_importances
            categorical_plot
            numerical_plot
        """
        
    def check_null(self, data):
        """This method returns a fraction of the
            null values per feature in the data
        ** Args: Data - pandas dataframe
        ** Return: decimal value - (sum of null values per feature / sum of data points)
        """
        
        nan_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
        print(f"Shape: {data.shape}, Number of Columns with NaN: {len(nan_cols)}")
        return data[nan_cols].isnull().sum()/data.shape[0]

    def plot_feature_importances(self, model, model_name, data, num_features=50):
        """Returns a plot of the feature importance as scored by the model
        ** Args: Data - pandas dataframe
                 Model - Algorithm
        ** Return: bar plot
        """
        plt.figure(figsize=(15, 30));
        feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['Importance'])
        feature_importance_df['Feature'] = data.columns
        sns.barplot(x="Importance", y="Features", 
                    data=feature_importance_df.sort_values(by=['Importance'], 
                    ascending = False).head(num_features))
        plt.title(model_name)
        plt.show();


    def categorical_plot(self, data, hue, cols=None):
        """Return a plot of categorical features in a data
        ** Args: Data - pandas dataframe
                 Hue - string
                 Categorical Columns - list
        ** Return: bar plot
        """
        if cols == None: cols = [cname for cname in data.columns if data[cname].dtype == 'object' and data[cname].nunique() < 20]
        for col in cols: 
            if col in data.columns:
                sns.countplot(y=col, hue=hue, data=data)
                plt.show()
                
    def numerical_plot(self, data, hue, cols=None, kde=True):
        """Return a plot of numerical features in a data
        ** Args: Data - pandas dataframe
                 Hue - string
                 cols - features list
        ** Return: bar plot
        """
        if cols == None: cols = [cname for cname in data.columns if data[cname].dtype in ['int', 'float']]
        for col in cols: 
            if col in data.columns:
                sns.displot(x=col, multiple='stack', kde=True, hue=hue, data=data)
                plt.title(col + 'Distribution')
                plt.show()