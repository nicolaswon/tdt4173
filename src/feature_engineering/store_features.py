import pandas as pd
from scipy.spatial.distance import cdist

def store_count(X, agg_cols, agg_name, **kwargs):
    return X.groupby(agg_cols).size().reset_index(name=agg_name, drop=False)

def average_revenue(X, agg_cols, agg_name, **kwargs):
    return  X.groupby(agg_cols)['revenue'].agg('mean').reset_index(name=agg_name, drop=False)

def total_revenue(X, agg_cols, agg_name, **kwargs):
    return  X.groupby(agg_cols)['revenue'].agg('sum').reset_index(name=agg_name, drop=False)

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
            
            number_same = compare_df[(compare_df['store_id'].isin(nearby_stores)) & (
                compare_df[store_type_group] == index_type)]['store_id'].count()
            
            store_count[index] = number_same
        
        df = pd.DataFrame.from_dict(store_count, orient='index', columns=["count"])
        df.index.rename('store_id', inplace=True)
        return df.reset_index()

def store_closest(stores_df, compare_df, store_type_group="lv4_desc"):
    """
    Id and distance of the closest store of same type in the same group.
    """
    
    store_types_in_group = stores_df[store_type_group].unique()
    df_list = []
    for store_type in store_types_in_group:
        stores_by_type = stores_df[stores_df[store_type_group] == store_type]
        stores_comp_by_type = compare_df[compare_df[store_type_group] == store_type]
        
        mat = cdist(stores_by_type[['lat', 'lon']], stores_comp_by_type[['lat', 'lon']], metric='euclidean')
        
        df = pd.DataFrame(
            mat, index=stores_by_type['store_id'], columns=stores_comp_by_type['store_id'])
        
        df = df[df > 0]
        
        stores = df.index
        closest = df.idxmin(axis=1)
        distance = df.min(axis=1)
        
        new_df = pd.DataFrame({'store_id': stores.values, 'closest_store': closest.values, 'distance': distance.values})
        df_list.append(new_df)
        
    
    return pd.concat(df_list, ignore_index=True)


def store_closest_by_store_groups(stores_df, compare_df, store_type_groups):
    df_list = []
    
    for store_type_group in store_type_groups:
        df = store_closest(stores_df, compare_df, store_type_group=store_type_group)
        df.rename(columns={'distance': f'distance_to_{store_type_group}'}, inplace=True)
        df_list.append(df[['store_id', f'distance_to_{store_type_group}']])

    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1).reset_index()