import pandas as pd
from scipy.spatial.distance import cdist

# Utility functions
def bus_stops_lat_lon(bus_stops_df):
    """
    Extract latitude and longitude as separate columns.
    """
    bus_stops_df['lng_lat'] = bus_stops_df['geometry'].str.extract(
        r'\((.*?)\)')
    bus_stops_df[['lon', 'lat']] = bus_stops_df['lng_lat'].str.split(
        " ", 1, expand=True)
    bus_stops_df[['lon', 'lat']] = bus_stops_df[[
        'lon', 'lat']].apply(pd.to_numeric)
    return bus_stops_df[['busstop_id', 'stopplace_type', 'importance_level', 'side_placement', 'geometry', 'lat', 'lon']]

def bus_stops_closest(stores_df, bus_stops_df, importance_level="Regionalt knutepunkt"):
    """
    Id and distance of the closest bus stop to all stores.
    """
    bus_stops_df = bus_stops_df[bus_stops_df['importance_level'] == importance_level]
    mat = cdist(stores_df[['lat', 'lon']],
                bus_stops_df[['lat', 'lon']], metric='euclidean')

    new_df = pd.DataFrame(
        mat, index=stores_df['store_id'], columns=bus_stops_df['busstop_id'])

    stores = stores_df.store_id
    closest = new_df.idxmin(axis=1)
    distance = new_df.min(axis=1)

    return pd.DataFrame({'store_id': stores.values, 'closest_bus_stop': closest.values, 'distance': distance.values})

def bus_stops_in_radius(stores_df, bus_stops_df, radius=0.1, importance_level=None):
    """
    Number of bus stops within a given radius. The importance level of bus stops can be specified.
    """
    if importance_level is not None:
        bus_stops_df = bus_stops_df[bus_stops_df['importance_level'] == importance_level]

    mat = cdist(stores_df[['lat', 'lon']],
                bus_stops_df[['lat', 'lon']], metric='euclidean')
    new_df = pd.DataFrame(
        mat, index=stores_df['store_id'], columns=bus_stops_df['busstop_id'])
    count = pd.DataFrame(new_df[new_df < radius].count(axis=1)).reset_index()
    count.rename(columns={0: 'count'}, inplace=True)
    return count

# Relevant feature engineering functions.
def bus_stops_distance_by_importance(stores_df, bus_stops_df, stop_importance_levels):
    """
    Distance for each store to the closest bus stop of each importance_level
    """
    df_list = []
    for importance_level in stop_importance_levels:
        importance_level_cleaned = importance_level.lower().replace(" ", "_")
        df = bus_stops_closest(stores_df, bus_stops_df, importance_level=importance_level)
        df.rename(columns={'distance': f'distance_to_{importance_level_cleaned}'}, inplace=True)
        df_list.append(df[['store_id', f'distance_to_{importance_level_cleaned}']])

    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)

def bus_stops_in_radius_by_importance(stores_df, bus_stops_df, stop_importance_levels, radius=0.01):
    """
    Number of bus stops in radius of store for each importance level.
    """
    df_list = []
    df_list.append(bus_stops_in_radius(stores_df, bus_stops_df, radius=radius).rename(columns={'count':'number_of_all_stop_types'})) # All bus stops in radius
    
    for importance_level in stop_importance_levels:
        importance_level_cleaned = importance_level.lower().replace(" ", "_")
        df = bus_stops_in_radius(stores_df, bus_stops_df, importance_level=importance_level, radius=radius)
        df.rename(columns={'count': f'number_of_{importance_level_cleaned}'}, inplace=True)
        df_list.append(df[['store_id', f'number_of_{importance_level_cleaned}']])

    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)