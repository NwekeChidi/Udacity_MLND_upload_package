import pandas as pd
import numpy as np
from .data_preprocessor import Preprocessor


class ModelEngineering(Preprocessor):
    def __init__(self):
        super().__init__()
        """Class for scoring base models.
     
        Attribute: None
        Methods:
            evaluate
            get_score
        """    
    
    def evaluate(predictor, test_features, test_labels, verbose=True):
        """
        Evaluate a model on a test set given the prediction endpoint.  
        Return binary classification metrics.
        :param predictor: A prediction endpoint
        :param test_features: Test features
        :param test_labels: Class labels for test data
        :param verbose: If True, prints a table of all performance metrics
        :return: A dictionary of performance metrics.
        """
        import numpy as np
        # Getting predictions   
        test_preds = predictor.predict(test_features)


        # calculate true positives, false positives, true negatives, false negatives
        tp = np.logical_and(test_labels, test_preds).sum()
        fp = np.logical_and(1-test_labels, test_preds).sum()
        tn = np.logical_and(1-test_labels, 1-test_preds).sum()
        fn = np.logical_and(test_labels, 1-test_preds).sum()

        # calculate binary classification metrics
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        # printing a table of metrics
        if verbose:
            print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
            print("\n{:<11} {:.3f}".format('Recall:', recall))
            print("{:<11} {:.3f}".format('Precision:', precision))
            print("{:<11} {:.3f}".format('Accuracy:', accuracy))
            print()

        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
                'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
    
    def get_score(self, data, target=None, model=None, objective='classification', ml_type='Supervised', prefit=False, scale=False, seed=42,   scale_type='StandardScaler', decompose=False, decompose_type='pca'):
        """Prints the score of a base model
        ** Args:
            model (object) default(None)- already defined model
            target (pandas.DataFrame) - target label for supervised learning
            data (pandas.DataFrame) - dataset
            objective (str) {"classification", "regression"} default('classification')- learning objective
            ml_type (str) {"supervised", "unsupervised"} default("Supervised") - learning type
            scale (bool) default(False)- if to scale or not to scale
            prefit (bool) - if model is already trained
            seed (int) - parameter for score reproduction
            scale_type (str) {'StandardScaler', 'RobustScaler', MinMaxScaler'} default('StandardScaler') - type of scaler to be used
            decompose (bool) default(False) - if to decompose or not, mostly used for unsupervised learning
            decompose_type (str) {'pca', 'nmf'} default('pca')- type of decomposition to be used
        ** 
        """
        
        
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.cluster import KMeans
        
        
        if ml_type.lower() == "Supervised":
            x_t, x_v, y_t, y_v = self.split_data(data, target)

            # scaling if required
            if scale:
                x_t, x_v, y_t, y_v = self.scale(data, target, scale_type)

            # instantiating the model
            objective = objective.lower()
            if model == None and objective == 'classification':
                model = LogisticRegression().fit(x_t, y_t)
            elif model == None and objective == 'regression':
                model = LinearRegression().fit(x_t, y_t)
                
            # training the model
            if not prefit:
                model.fit(x_t, y_t)
                
            # getting model scores
            train_score = model.score(x_t, y_t)
            test_score = model.score(x_v, y_v)
            print(f"Train Score: {train_score:.4f},    Test Score: {test_score:.4f}")
        else:
            # splitting the data
            x_t, x_v = self.split_data(data, ml_type='unsupervised')
            
            # scaling if required
            if scale:
                x_t, x_v = self.scale(data, scale_type, ml_type='unsupervised')
            
            # decomposing if required
            if decompose:
                x_t, x_v = self.decompose(data, decompose_type)
                
            # instantiating the model
            if model == None:
                model = KMeans()
            
            # training the model
            if not prefit:
                model.fit(x_t)
                
            # getting model scores
            train_score = model.score(x_t)
            test_score = model.score(x_v)
            print(f"Train Score: {train_score:.4f},    Test Score: {test_score:.4f}")
    
    def fill_and_score(self, data, target, fill_w=['mean', 'median', 'std', 0, -999]):
        for i in range(len(fill_w)):
            print('filling null with: ', fill_w[i], '.......')
            cols = data.columns
            for ii in cols:
                if fill_w[i] == 'mean':
                    data[ii] = data[ii].fillna(data[ii].mean())
                elif fill_w[i] == 'median':
                    data[ii] = data[ii].fillna(data[ii].median())
                elif fill_w[i] == 'std':
                    data[ii] = data[ii].fillna(data[ii].std())
                else:
                    data[ii] = data[ii].fillna(fill_w[i])
            print('Done filling null!')
            print('Getting score ......')
            self.get_score(data, target)
        