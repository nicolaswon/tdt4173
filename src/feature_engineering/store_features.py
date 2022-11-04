import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def store_count(X, agg_cols, agg_name, **kwargs):
    return X.groupby(agg_cols).size().reset_index(name=agg_name, drop=False)

def average_revenue(X, agg_cols, agg_name, **kwargs):
    return  X.groupby(agg_cols)['revenue'].agg('mean').reset_index(name=agg_name, drop=False)

def total_revenue(X, agg_cols, agg_name, **kwargs):
    return  X.groupby(agg_cols)['revenue'].agg('sum').reset_index(name=agg_name, drop=False)

    

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

class AggTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, agg_cols, agg_name, calculations, stores_extra = None, sample_revenue=None):
        self.agg_cols = agg_cols
        self.agg_name = agg_name
        self.calculations = calculations
        self.sample_revenue = sample_revenue
        self.stores_extra = stores_extra
        
        
    def fit(self, X, y=None):
        if self.stores_extra is not None:
            X = pd.concat([X, self.stores_extra], ignore_index=True).drop_duplicates()
        
        if self.sample_revenue is not None:
            X['revenue'] = self.sample_revenue
        
        mapping = self.calculations(X, self.agg_cols, self.agg_name)
        self.mapping_ = mapping
        
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        for index, row in self.mapping_.iterrows():
            ind = (X[self.agg_cols] == row[self.agg_cols]).all(axis=1)
            X.loc[ind, self.agg_name] = row[self.agg_name]
            
        return pd.DataFrame(X[self.agg_name])
    
    def get_feature_names_out(self, names=None):
        return [self.agg_name]
    
def stores_in_radius(stores_df, compare_df, radius=0.1, store_type_group=None):
    mat = cdist(stores_df[['lat', 'lon']],
                compare_df[['lat', 'lon']], metric="euclidean")
    
    new_df = pd.DataFrame(
        mat, index=stores_df['store_id'], columns=compare_df['store_id']
    )
    
    if store_type_group is None:
        count = new_df[(new_df < radius) & (new_df > 0)].count(axis=1)
        return count.to_frame(name="count").reset_index()
    
    else:
        test_df = new_df[(new_df < radius) & (new_df > 0)]
        store_count = {}
        
        for index, row in test_df.iterrows():
            nearby_stores = row.dropna().index.values
            index_type = compare_df.loc[compare_df['store_id'] == index, store_type_group].iat[0]
            
            # compare_df[compare_df['store_id']
            #                         == index][store_type_group].values[0]
            
            number_same = compare_df[(compare_df['store_id'].isin(nearby_stores)) & (
                compare_df[store_type_group] == index_type)]['store_id'].count()
            
            store_count[index] = number_same
        
        df = pd.DataFrame.from_dict(store_count, orient='index', columns=["count"])
        df.index.rename('store_id', inplace=True)
        return df.reset_index()
    
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