from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Preprocessor:
    """Class for preprocessing data.
     
        Attribute: None
        Methods:
            decompose
            scale
            split_data
            fit_decomposer
            fit_scaler
        """
    
    def decompose(self, data, decompose_type='pca', n_components=None):
        """This method splits and reduces the dimension of the data used a preferrable decomposer
        ** Args:
            data (pandas.DataFrame) - dataset
            decompose_type (str) {'pca', 'nmf'} default(nmf) - decomposing algorithm
            n_components (int) default(None) - number of components
        ** Return:
            decomposed_x_train, decomposed_x_valid
        """
        from sklearn.decomposition import PCA, NMF
        
        # getting n_components
        if n_components == None:
            n_components = data.shape[1] - 1
            
        # splitting the data
        x_t, x_v = self.split_data(data, ml_type='Unsupervised')
            
        # instantiating the decomposer
        pca = PCA(n_components)
        nmf = NMF(n_components)
        
        d_t = decompose_type.lower()
        if d_t == 'pca':
            x_t, x_v = self.fit_decomposer(pca, x_t, x_v)
        else:
            x_t, x_v = self.fit_decomposer(nmf, x_t, x_v)
            
        return x_t, x_v
           
    def split_data(self, data, target=None, ml_type='Supervised', seed=42):
        """This method splits the data into train and test sets
        ** Args:
            data (pandas.DataFrame) - dataset
            target (pandas.DataFrame) - target label for supervised learning
            ml_type (str) {"supervised", "unsupervised"} default("Supervised") - learning type
            seed (int) default(42) - random integer for reproducibility
        ** Return:
            split data with their corresponding labels if applicable
        """
        
        if ml_type.lower() == 'supervised':
            x_t, x_v, y_t, y_v = train_test_split(data, target, stratify=target, test_size=0.3, random_state=seed)
            return x_t, x_v, y_t, y_v
        else:
            train_idx = round(0.7*data.shape[0])
            x_t = data[:train_idx]
            x_v = data[train_idx:]
            return x_t, x_v
    

    def scale(self, data, target=None, scale_type='StandardScaler', ml_type='Supervised'):
        """This method splits the data into train and test sets
        ** Args:
            data (pandas.DataFrame) - dataset
            target (pandas.DataFrame) - target label for supervised learning
            ml_type (str) {"supervised", "unsupervised"} default("Supervised") - learning type
            scale_type (str) {'StandardScaler', 'RobustScaler', MinMaxScaler'} default('StandardScaler') - type of scaler to be used
        ** Return:
            scaled split data
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        from sklearn.model_selection import train_test_split
        
        # splitting the data
        x_t, x_v, y_t, y_v = self.split_data(data, target)
        
        # instantiating the scaler
        ss = StandardScaler()
        mm = MinMaxScaler()
        rs = RobustScaler()
        
        # scaling 
        scale_type = scale_type.lower()
        if ml_type.lower() == 'supervised':
            if scale_type == 'standardscaler':
                x_t, x_v = self.fit_scaler(ss, x_t, x_v)
            if scale_type == 'robustscaler':
                x_t, x_v = self.fit_scaler(rs, x_t, x_v)
            if scale_type == 'minmaxscaler':
                x_t, x_v = self.fit_scaler(mm, x_t, x_v)
            return x_t, x_v, y_t, y_v
        else:
            if scale_type == 'StandardScaler':
                x_t, x_v = self.fit_scaler(ss, x_t, x_v)
            if scale_type == 'RobustScaler':
                x_t, x_v = self.fit_scaler(rs, x_t, x_v)
            else:
                x_t, x_v = self.fit_scaler(mm, x_t, x_v)
            return x_t, x_v
    
    def fit_scaler(self, scaler, x_train, x_val):
        """This method scales the train and test data using a prefferable 
           scaling algorithm
        ** Args:
            scaler (obj) - scaling algorithm to be used
            x_train (ndarray) - train data
            x_val (ndarray) - test data
        ** Return:
            scaled x_train and x_val
        """
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        return x_train, x_val
    
    def fit_decomposer(self, decomposer, x_train, x_val):
        """This method reduces the dimension of the train and test data using a prefferable 
           dimensionality reduction algorithm
        ** Args:
            decomposer (obj) - dimensionality reduction algorithm to be used
            x_train (ndarray) - train data
            x_val (ndarray) - test data
        ** Return:
            decomposed x_train and x_val
        """
        decomposer.fit(x_train)
        x_train = decomposer.transform(x_train)
        x_val = decomposer.transform(x_val)
        return x_train, x_val