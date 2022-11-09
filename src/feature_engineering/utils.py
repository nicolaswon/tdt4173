import pandas as pd
from feature_engineering.bus_stop_features import *
from feature_engineering.demographic_features import *
from feature_engineering.store_features import *

def enrich_keys(stores_df, raw_path):
    grunnkrets_norway_df = set_year_2016(pd.read_csv(f"{raw_path}/grunnkrets_norway_stripped.csv"))
    plaace_df = pd.read_csv(f"{raw_path}/plaace_hierarchy.csv")
    
    return stores_df.merge(grunnkrets_norway_df, on="grunnkrets_id", how="left").merge(plaace_df, on="plaace_hierarchy_id", how="left")

def set_year_2016(dataframe):
    return dataframe[dataframe['year'] == 2016].drop(['year'], axis=1)

def combine_keys(dataframe):
    dataframe = dataframe.copy()
    dataframe['t_district'] = dataframe['district_name'] + dataframe['municipality_name']
    return dataframe

def data_enricher(stores_df, raw_path, geo_groups, importance_levels):
    stores_df = set_year_2016(stores_df)
    bus_stops_df = bus_stops_lat_lon(pd.read_csv(f"{raw_path}/busstops_norway.csv"))
    grunnkrets_age_df = set_year_2016(pd.read_csv(f"{raw_path}/grunnkrets_age_distribution.csv"))
    grunnkrets_household_pop_df = set_year_2016(pd.read_csv(f"{raw_path}/grunnkrets_households_num_persons.csv"))
    grunnkrets_household_inc_df = set_year_2016(pd.read_csv(f"{raw_path}/grunnkrets_income_households.csv"))
    grunnkrets_norway_df = combine_keys(set_year_2016(pd.read_csv(f"{raw_path}/grunnkrets_norway_stripped.csv")))
    plaace_df = pd.read_csv(f"{raw_path}/plaace_hierarchy.csv")
    
    # Merged dataframe for simple joining.
    merged_df = stores_df.merge(grunnkrets_norway_df, on="grunnkrets_id", how="left").merge(plaace_df, on="plaace_hierarchy_id", how="left")
    
    # Population cont for all geographic groups
    population_df = population_count_grouped_by_geo_group(stores_df, grunnkrets_age_df, grunnkrets_norway_df, geo_groups)
    
    # Population density for all geographic groups
    population_density_df = population_density_grouped_by_geo_group(stores_df, grunnkrets_age_df, grunnkrets_norway_df, geo_groups)
    
    # Age distribution for all geographic groups
    age_dist_df = age_dist_by_geo_group(stores_df, grunnkrets_age_df, grunnkrets_norway_df, geo_groups)
    
    # Household distribution for all geographic groups
    household_dist_df = household_dist_by_geo_group(stores_df, grunnkrets_household_pop_df, grunnkrets_norway_df, geo_groups)
    
    # Mean income for all geographic groups
    mean_income_df = mean_income_per_capita_by_geo_group(stores_df, grunnkrets_age_df, grunnkrets_household_inc_df, grunnkrets_norway_df, geo_groups)
    
    bus_stops_in_radius = bus_stops_in_radius_by_importance(stores_df, bus_stops_df, importance_levels, radius=0.1)
    
    bus_stops_distances = bus_stops_distance_by_importance(stores_df, bus_stops_df, importance_levels)
    
    merged_df = (merged_df
                 .merge(population_df, on="store_id", how="left")
                 .merge(population_density_df, on="store_id", how="left")
                 .merge(age_dist_df, on="store_id", how="left")
                 .merge(household_dist_df, on="store_id", how="left")
                 .merge(mean_income_df, on="store_id", how="left")
                 .merge(bus_stops_in_radius, on="store_id", how="left")
                 .merge(bus_stops_distances, on="store_id", how="left")
    )
    
    return merged_df