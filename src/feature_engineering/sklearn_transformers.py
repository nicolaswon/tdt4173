import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from feature_engineering.store_features import *

def is_name(function_transformer, feature_names_in):
    return ["is"]  # feature names out

def is_null(x):
    return x.notna()

def is_null_pipeline():
    return make_pipeline(
        FunctionTransformer(is_null, feature_names_out=is_name)
        )
    
def one_hot_encode_pipeline():
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse=False)
    )
    
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None, sample_weight=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.sample_weight = sample_weight
        
    def fit(self, X, y=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=self.sample_weight['revenue'])
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f'Cluster {i} similarity' for i in range(self.n_clusters)]
    
    
class StoresInRadiusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radius, store_type_group=None, stores_extra=None):
        self.radius = radius
        self.store_type_group = store_type_group
        self.stores_extra = stores_extra
        
    def fit(self, X, y=None):
        if self.stores_extra is not None:
            self.geo_map_ = pd.concat([X, self.stores_extra], ignore_index=True).drop_duplicates()
        else:
            self.geo_map_ = X
            
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        df = stores_in_radius(X, self.geo_map_, radius=self.radius, store_type_group=self.store_type_group)
        return pd.DataFrame(X.merge(df, on="store_id", how="left")[f'count'])
    
    def get_feature_names_out(self, names=None):
        return ['in_radius']
    
    
class AggTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, agg_cols, agg_name, calculations, stores_extra = None, sample_revenue=None):
        self.agg_cols = agg_cols
        self.agg_name = agg_name
        self.calculations = calculations
        self.sample_revenue = sample_revenue
        self.stores_extra = stores_extra
        
        
    def fit(self, X, y=None):
        X = X.copy()[['store_id'] + self.agg_cols]
        
        if self.stores_extra is not None:
            self.stores = pd.concat([X, self.stores_extra[['store_id']+ self.agg_cols]], ignore_index=True).drop_duplicates()
        else:
            self.stores = X
        
        if self.sample_revenue is not None:
            self.stores['revenue'] = self.sample_revenue['revenue']
        
        return self
    
    def transform(self, X, y=None):
        X = X.copy()[['store_id'] + self.agg_cols]
        
        if self.sample_revenue is None:
            mapping = self.calculations(pd.concat([X, self.stores], ignore_index=True).drop_duplicates(), self.agg_cols, self.agg_name)
        else:
            mapping = self.calculations(self.stores, self.agg_cols, self.agg_name)
        
        for index, row in mapping.iterrows():
            ind = (X[self.agg_cols] == row[self.agg_cols]).all(axis=1)
            X.loc[ind, self.agg_name] = row[self.agg_name]
            
        return pd.DataFrame(X[self.agg_name])
    
    def get_feature_names_out(self, names=None):
        return [self.agg_name]
    

class ClosestStore(BaseEstimator, TransformerMixin):
    def __init__(self, stores_extra=None, store_type_groups=['lv1_desc']):
        self.stores = None
        self.store_type_groups = store_type_groups
        self.stores_extra = stores_extra
        
    def fit(self, X, y=None):
        X = X.copy()[['store_id', 'lat', 'lon']+self.store_type_groups]
        if self.stores_extra is not None:
            self.stores = pd.concat([X, self.stores_extra[['store_id', 'lat', 'lon']+self.store_type_groups]], ignore_index=True)
        else:
            self.stores = X
            
        return self
    
    def transform(self, X, y=None):
        X = X.copy()[['store_id', 'lat', 'lon']+self.store_type_groups]
        
        combined_df = pd.concat([X, self.stores], ignore_index=True)
        
        self.closest = store_closest_by_store_groups(X, combined_df, self.store_type_groups)
        
        return X.merge(self.closest, on="store_id", how="left").iloc[:,-len(self.store_type_groups):]
    
    def get_feature_names_out(self, names=None):
        return [f"distance_{store_type}" for store_type in self.store_type_groups]