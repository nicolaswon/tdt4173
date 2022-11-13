import pandas as pd
import numpy as np

def population(dataset_age):
    population = dataset_age.drop(["grunnkrets_id"], axis=1).sum(axis=1)
    dataset_age["population_count"] = population
    return dataset_age[["grunnkrets_id", "population_count"]]

def population_grouped(data_age, data_geography, grouping_element):
    age_df = population(data_age)
    geography_df = data_geography
    population_df = age_df.merge(geography_df, how="left", on="grunnkrets_id")
    grouped_df = population_df.groupby([grouping_element], as_index=False)["population_count"].sum()
    return grouped_df

def population_count_grouped_by_geo_group(stores_df, age_df, grunnkrets_df, geo_groups): 
    combined_df = stores_df.merge(grunnkrets_df, how = "left", on = "grunnkrets_id")

    population_columns = ["population_count"]
    df_list = []

    for geo_group in geo_groups: 
        pop_df = population_grouped(age_df, grunnkrets_df, geo_group)
        merged_df = combined_df.merge(pop_df, how = "left", on = geo_group)[["store_id"] + population_columns]
        merged_df.set_index("store_id", inplace = True)
        merged_df2 = merged_df.add_prefix(f'{geo_group}_')
        df_list.append(merged_df2)

    return pd.concat(df_list, axis = 1).reset_index()

def population_density(age_df, geo_df, grouping_element):
    age_data = population(age_df)
    geo_df = geo_df
    combined_df = age_data.merge(geo_df, how="left", on="grunnkrets_id")
    density_df = combined_df.groupby([grouping_element], as_index=False)[
        ["population_count", "area_km2"]].sum()
    density_df["density"] = density_df["population_count"] / \
        density_df["area_km2"]
    return density_df

def population_density_grouped_by_geo_group(stores_df, age_df, grunnkrets_df, geo_groups):
    grunnkrets_df_2016 = grunnkrets_df
    combined_df = stores_df.merge(grunnkrets_df_2016, how = "left", on = "grunnkrets_id")

    pop_density_columns = ["density"]
    df_list = []

    for geo_group in geo_groups: 
        pop_df = population_density(age_df, grunnkrets_df, geo_group)
        merged_df = combined_df.merge(pop_df, how = "left", on = geo_group)[["store_id"] + pop_density_columns]
        merged_df.set_index("store_id", inplace = True)
        merged_df2 = merged_df.add_prefix(f'{geo_group}_')
        df_list.append(merged_df2)

    return pd.concat(df_list, axis = 1).reset_index()

def age_distrubution(grunnkrets_age_df, geographic_df, grouping_element):
    age_df1 = grunnkrets_age_df
    age_df1["num_kids"] = age_df1.iloc[:, 1:8].sum(axis=1)
    age_df1["num_kids+"] = age_df1.iloc[:, 8:14].sum(axis=1)
    age_df1["num_youths"] = age_df1.iloc[:, 14: 19].sum(axis=1)
    age_df1["num_youthAdult"] = age_df1.iloc[:, 19:27].sum(axis=1)
    age_df1["num_adult"] = age_df1.iloc[:, 27:37].sum(axis=1)
    age_df1["num_adults+"] = age_df1.iloc[:, 37:62].sum(axis=1)
    age_df1["num_pensinors"] = age_df1.iloc[:, 62:92].sum(axis=1)

    age_df2 = age_df1[["grunnkrets_id", "num_kids", "num_kids+", "num_youths",
                       "num_youthAdult", "num_adult", "num_adults+", "num_pensinors"]]

    pop_df = population(grunnkrets_age_df)
    new_geo_df = geographic_df.drop(["geometry", "area_km2"], axis=1)
    combined_df = age_df2.merge(pop_df, how="inner", on="grunnkrets_id").merge(
        new_geo_df, how="inner", on="grunnkrets_id")
    list_columns = ["num_kids", "num_kids+", "num_youths",
                    "num_youthAdult", "num_adult", "num_adults+", "num_pensinors"]
    combined_df2 = combined_df.groupby([grouping_element], as_index=False)[
        list_columns].sum()

    pop_gk = population_grouped(
        grunnkrets_age_df, geographic_df, grouping_element)
    new_df = combined_df2.merge(pop_gk, how="inner", on=grouping_element)

    new_df["kids_%"] = new_df["num_kids"] / new_df["population_count"]
    new_df["kids+_%"] = new_df["num_kids+"] / new_df["population_count"]
    new_df["youths_%"] = new_df["num_youths"] / new_df["population_count"]
    new_df["youthAdult_%"] = new_df["num_youthAdult"] / \
        new_df["population_count"]
    new_df["adult_%"] = new_df["num_adult"] / new_df["population_count"]
    new_df["adults+_%"] = new_df["num_adults+"] / new_df["population_count"]
    new_df["pensinors_%"] = new_df["num_pensinors"] / \
        new_df["population_count"]

    age_dist_df = new_df.drop(["population_count"]+["num_kids", "num_kids+", "num_youths",
                       "num_youthAdult", "num_adult", "num_adults+", "num_pensinors"], axis=1)
    return age_dist_df

def age_dist_by_geo_group(stores_df, age_df, grunnkrets_norway_df, geo_groups): 
    combined_df = stores_df.merge(grunnkrets_norway_df, how = "left", on = "grunnkrets_id")

    age_columns = ['kids_%', 'kids+_%', 'youths_%',
       'youthAdult_%', 'adult_%', 'adults+_%', 'pensinors_%']

    df_list = []
    for geo_group in geo_groups: 
      age_dist_df = age_distrubution(age_df, grunnkrets_norway_df, geo_group)
      merged_df = combined_df.merge(age_dist_df, how = "left", on = geo_group)[["store_id"] + age_columns]
      merged_df.set_index("store_id", inplace = True)
      merged_df2 = merged_df.add_prefix(f'{geo_group}_')
      df_list.append(merged_df2)
    
    return pd.concat(df_list, axis = 1).reset_index()

def household_type_distrubution(grunnkrets_norway_df, grunnkrets_household_pop_df, grouping_element):
    combined_df = grunnkrets_norway_df.merge(grunnkrets_household_pop_df, how="inner", on="grunnkrets_id")

    list_columns = ["couple_children_0_to_5_years", "couple_children_18_or_above", "couple_children_6_to_17_years",
                    "couple_without_children", "single_parent_children_0_to_5_years", "single_parent_children_18_or_above",
                    "single_parent_children_6_to_17_years", "singles"]

    grouped_df = combined_df.groupby([grouping_element], as_index=False)[
        list_columns].sum()
    grouped_df["tot_pop_count"] = grouped_df.iloc[:, 1:].sum(axis=1)

    grouped_df["%_dist_of_couple_children_0_to_5_years"] = grouped_df["couple_children_0_to_5_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_children_18_or_above"] = grouped_df["couple_children_18_or_above"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_children_6_to_17_years"] = grouped_df["couple_children_6_to_17_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_couple_without_children"] = grouped_df["couple_without_children"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_0_to_5_years"] = grouped_df["single_parent_children_0_to_5_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_18_or_above"] = grouped_df["single_parent_children_18_or_above"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_single_parent_children_6_to_17_years"] = grouped_df["single_parent_children_6_to_17_years"] / \
        grouped_df["tot_pop_count"]
    grouped_df["%_dist_of_singles"] = grouped_df["singles"] / \
        grouped_df["tot_pop_count"]

    returned_df = grouped_df.drop(["tot_pop_count"], axis=1)
    return returned_df

def household_dist_by_geo_group(stores_df, grunnkrets_household_pop_df, grunnkrets_norway_df, geo_groups):
    combined_df = stores_df.merge(grunnkrets_norway_df, how = "left", on = "grunnkrets_id")
    
    household_columns = ['couple_children_0_to_5_years', 'couple_children_18_or_above', 'couple_children_6_to_17_years', 'couple_without_children',
       'single_parent_children_0_to_5_years','single_parent_children_18_or_above','single_parent_children_6_to_17_years', 'singles',
       '%_dist_of_couple_children_0_to_5_years','%_dist_of_couple_children_18_or_above','%_dist_of_couple_children_6_to_17_years',
       '%_dist_of_couple_without_children','%_dist_of_single_parent_children_0_to_5_years','%_dist_of_single_parent_children_18_or_above',
       '%_dist_of_single_parent_children_6_to_17_years', '%_dist_of_singles']
       
    df_list = []

    for geo_group in geo_groups: 
        household_type_df = household_type_distrubution(grunnkrets_norway_df, grunnkrets_household_pop_df, geo_group)
        merged_df = combined_df.merge(household_type_df, how = "left", on = geo_group)[["store_id"] + household_columns]
        merged_df.set_index("store_id", inplace = True)
        merged_df2 = merged_df.add_prefix(f'{geo_group}_')
        df_list.append(merged_df2)
    return pd.concat(df_list, axis = 1)

def mean_income_per_capita(grunnkrets_age_df, grunnkrets_household_inc_df):
    "mean income per capita per grunnkrets"
    age_df = population(grunnkrets_age_df)
    age_and_income_df = age_df.merge(grunnkrets_household_inc_df, how='left', on='grunnkrets_id')
    mean_income = age_and_income_df.drop(['singles', 'couple_without_children',
                                         'couple_with_children', 'other_households', 'single_parent_with_children'], axis=1)
    mean_income['mean_income'] = mean_income['all_households'] / \
        mean_income['population_count']
    mean_income = mean_income.drop(['all_households'], axis=1)

    return mean_income

def mean_income_per_capita_grouped(grunnkrets_age_df, grunnkrets_household_inc_df, grunnkrets_norway_df, geo_group, agg_name):
    # gets data from mean_income_per_capita functino
    data_mean_income = mean_income_per_capita(grunnkrets_age_df, grunnkrets_household_inc_df)
    # gets data from geography set and makes sure we only use data for 2016
    # gets the data of mean income with the geography data
    mean_income_geo_df = data_mean_income.merge(
        grunnkrets_norway_df, how='left', on='grunnkrets_id')
    # sum the number of people based on grouping element
    grouped_population_df = mean_income_geo_df.groupby(
        [geo_group], as_index=False)["population_count"].sum()
    # merge this with the grunnkrets to see both total population per selected area and grunnkrets
    total_grouped_df = mean_income_geo_df.merge(
        grouped_population_df, how='left', on=geo_group)
    portion_income_df = total_grouped_df
    # find ration of grunnkrets to total population and multiply this with grunnkrets mean income
    portion_income_df['mean_income'] = total_grouped_df['mean_income'] * \
        total_grouped_df['population_count_x'] / \
        total_grouped_df['population_count_y']
    # add these incomes together, should add up to the total mean income for the selected area
    grouped_income_df = portion_income_df.groupby(
        [geo_group])["mean_income"].sum().reset_index(name=agg_name)
    return grouped_income_df

def mean_income_per_capita_by_geo_group(stores_df, grunnkrets_age_df, grunnkrets_household_inc_df, grunnkrets_norway_df, geo_groups):
    merged_df = stores_df.merge(grunnkrets_norway_df, how="left", on="grunnkrets_id")
    
    df_list = []
    for geo_group in geo_groups:
        df = mean_income_per_capita_grouped(grunnkrets_age_df, grunnkrets_household_inc_df, grunnkrets_norway_df, geo_group, agg_name=f'{geo_group}_mean_income')
        df_list.append(merged_df.merge(df, how="left", on=[geo_group])[['store_id', f'{geo_group}_mean_income']])
    
    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1)

def num_households(household_dist):
    household_dist = household_dist.copy()
    population = household_dist.drop(["grunnkrets_id"], axis=1).sum(axis=1)
    household_dist["household_count"] = population
    return household_dist[["grunnkrets_id", "household_count"]]

def num_households_geo(geo_group, household_dist, grunnkrets_df):
    _num_households = num_households(household_dist)
    merged_df = grunnkrets_df.merge(_num_households, on="grunnkrets_id", how="inner")
    return merged_df.groupby([geo_group], as_index=False)['household_count'].sum()

def total_grunnkrets_income(income_dist, household_dist):
    _num_households = num_households(household_dist)
    merged_df = income_dist.merge(_num_households, on="grunnkrets_id", how="inner")[['grunnkrets_id', 'household_count', 'all_households']]
    merged_df['total_income'] = merged_df['household_count'] * merged_df['all_households']
    return merged_df[['grunnkrets_id', 'total_income']]    

def total_income_geo(geo_group, income_dist, household_dist, grunnkrets_df):
    grunnkrets_income = total_grunnkrets_income(income_dist, household_dist)
    merged_df = grunnkrets_df.merge(grunnkrets_income, on="grunnkrets_id", how="inner")
    return merged_df.groupby([geo_group], as_index=False)['total_income'].sum()

def average_household_income_geo(geo_group, income_dist, household_dist, grunnkrets_df):
    income = total_income_geo(geo_group, income_dist, household_dist, grunnkrets_df)
    households = num_households_geo(geo_group, household_dist, grunnkrets_df)
    
    merged_df = income.merge(households, on=geo_group, how="inner")
    merged_df[f'avg_household_income_{geo_group}'] = merged_df['total_income'] / merged_df['household_count']
    return merged_df[[geo_group, f'avg_household_income_{geo_group}']]
    
def average_household_income_by_geo_groups(stores_df, geo_groups, income_dist, household_dist, grunnkrets_df):
    merged_df = stores_df.merge(grunnkrets_df, how="left", on="grunnkrets_id")
    
    df_list = []
    for geo_group in geo_groups:
        df = average_household_income_geo(geo_group, income_dist, household_dist, grunnkrets_df)
        df_list.append(merged_df.merge(df, how="left", on=[geo_group])[['store_id', f'avg_household_income_{geo_group}']])
    
    dfs = [df.set_index('store_id') for df in df_list]
    return pd.concat(dfs, axis=1).reset_index()