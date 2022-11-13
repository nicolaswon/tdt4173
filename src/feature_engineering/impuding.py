from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

def impude_gk(stores_df, grunnkrets_df, nan_col):
    geo_df2 = grunnkrets_df
    merged_df = stores_df.merge(geo_df2, how = "left", on = "grunnkrets_id")
    NaN_df = merged_df[merged_df[nan_col].isna()]
    split_df = merged_df[merged_df[nan_col].notna()]

    mat = cdist(NaN_df[['lat', 'lon']],
                split_df[['lat', 'lon']], metric='euclidean')

    new_df = pd.DataFrame(mat, index= NaN_df['grunnkrets_id'], columns=split_df['grunnkrets_id'])

    grunnkrets_id = NaN_df.grunnkrets_id
    closest = new_df.idxmin(axis=1)
    distance = new_df.min(axis=1)

    closest_df_with_distance = pd.DataFrame({"grunnkrets_id": grunnkrets_id, "closest_valid_id" : closest.values, "distance": distance.values})
    closest_df = pd.DataFrame({"grunnkrets_id": grunnkrets_id, "closest_valid_id" : closest.values})

    df_with_values_from_valid_id = split_df[split_df["grunnkrets_id"].isin(closest.values)]
    df_with_values_from_valid_id_removed_duplicates = df_with_values_from_valid_id.drop_duplicates(subset = ["grunnkrets_id"])

    df_valid_geo_data = df_with_values_from_valid_id_removed_duplicates.iloc[:,11:]
    df_with_only_gk_id = df_with_values_from_valid_id_removed_duplicates[["grunnkrets_id"]]
    df_list = [df_with_only_gk_id,df_valid_geo_data ]
    df_valid_geo_data_and_id = pd.concat(df_list, axis=1)

    df_without_nan = NaN_df.iloc[:,:11]

    df_including_closest_valid_id = df_without_nan.reset_index().merge(closest_df, how = "left", on = "grunnkrets_id").set_index("index")

    df_impuded = df_including_closest_valid_id.reset_index().merge(df_valid_geo_data_and_id, how="left", left_on ="closest_valid_id", right_on="grunnkrets_id").set_index("index")
    df_impuded_without_duplicates = df_impuded.drop_duplicates(subset="store_id")

    new_df_impuded = df_impuded_without_duplicates.drop(["closest_valid_id", "grunnkrets_id_y"], axis = 1).rename(columns= {"grunnkrets_id_x": "grunnkrets_id"})


    impuded = pd.concat([split_df, new_df_impuded])

    return impuded