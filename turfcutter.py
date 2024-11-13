import pandas as pd
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import math
import pathlib
import simplekml
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import time
import click
from pyproj import Proj, transform, Transformer
from sklearn.cluster import DBSCAN
import geopandas as gpd
import contextily as ctx


API_KEY = '<your_key>'
GEOCODING_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
INDEX_COLS = ['Address Id', 'VANID']

ADDRESS_VALIDATION_URL = f'https://addressvalidation.googleapis.com/v1:validateAddress?key={API_KEY}'


def remove_middle_initials(name):
    return " ".join(word for word in name.split() if len(word) > 1)


def remove_selected_substrings(name, substrings_to_drop=None):
    substrings_to_drop = substrings_to_drop or ['Jr', 'Sr', 'iii', 'III', 'IV', 'iv']
    return " ".join(word for word in name.split() if word not in substrings_to_drop)


def clean_names(s):
    lastname_firstname = s.str.split(', ', expand=True)
    firstname_lastname = lastname_firstname[1] + ' ' + lastname_firstname[0]
    firstname_lastname = firstname_lastname.apply(remove_middle_initials)
    firstname_lastname = firstname_lastname.apply(remove_selected_substrings)
    return firstname_lastname


def longest_common_word_prefix(addresses):
    # Split all addresses into lists of words and find the common prefix across all lists
    common_words = list(zip(*[addr.split() for addr in addresses]))
    prefix = []
    for words in common_words:
        if all(word == words[0] for word in words):
            prefix.append(words[0])
        else:
            break
    return " ".join(prefix)


def floats_to_str(s: pd.Series, scaler: int = 1e5):
    return (s * scaler).astype(int).astype(str)


def validate_address(addr):
    # Structure the address for the API request
    address = {
        "address": {
            "regionCode": "US",
            "postalCode": str(addr['Zip/Postal']),
            "administrativeArea": addr['State/Province'],
            "locality": addr['City'],
            "addressLines": [addr['Street Address']]
        }
    }

    # Send request to Address Validation API
    response = requests.post(ADDRESS_VALIDATION_URL, json=address)
    result = response.json()

    if 'result' in result and 'geocode' in result['result']:
        validated_address = result['result']['address']
        try:
            corrected_street_address = validated_address['postalAddress']['addressLines'][0]
        except:
            corrected_street_address = addr['Street Address']
        latitude = result['result']['geocode']['location']['latitude']
        longitude = result['result']['geocode']['location']['longitude']

        return corrected_street_address, latitude, longitude
    else:
        print("Error:", result.get("error", "Unknown error"))
        return None, None, None


def manhattan_latlon_distance_to_feet(lat1, lon1, lat2, lon2):
    R = 20902640  # feet
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)
    distance_ns = R * dlat
    avg_lat = (lat1 + lat2) / 2
    distance_ew = R * dlon * math.cos(avg_lat)
    manhattan_distance_feet = distance_ns + distance_ew
    return manhattan_distance_feet


def get_distance_matrix_for_addresses(address_df, min_distance=10):
    res = defaultdict(dict)
    for addr1, v1 in address_df.iterrows():
        for addr2, v2 in address_df.iterrows():
            if addr1 == addr2:
                res[addr1][addr2] = 0
                continue
            approx_distance = manhattan_latlon_distance_to_feet(v1['latitude'], v1['longitude'],
                                                                v2['latitude'], v2['longitude'])
            effective_distance = max(min_distance, approx_distance)
            res[addr1][addr2] = effective_distance
    distance_matrix = pd.DataFrame.from_dict(res)
    return distance_matrix


# Function to get latitude and longitude from an address
def get_lat_long(address):
    params = {
        'address': address,
        'key': API_KEY
    }
    response = requests.get(GEOCODING_API_URL, params=params)
    result = response.json()

    if result['status'] == 'OK':
        location = result['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        print(f"Error geocoding {address}: {result['status']}")
        return None, None


def compute_affinity_from_distance(distance_matrix, gamma=1.0):
    # Convert distance matrix to affinity matrix using an RBF kernel (Gaussian similarity)
    affinity_matrix = np.exp(-gamma * distance_matrix ** 2)
    return affinity_matrix

def cluster_stats(labels, distance_matrix, info_df):
    stats_all_clusters = {}
    for i in sorted(labels.unique()):
        submatrix = distance_matrix.loc[labels == i, labels == i]
        stats = {'max_closest_neighbor': submatrix.replace(0, np.inf).min().max(),
                 'max_span': submatrix.max().max(),
                 'voter_count': info_df.loc[submatrix.index].shape[0],
                 'address_count': submatrix.shape[0]
                }
        stats_all_clusters[i] = stats
    return pd.DataFrame(stats_all_clusters)


def validate_clusters(labels, distance_matrix, info_df, closest_neighbor_lim, furthest_neighbor_lim,
                      max_addr, max_voters):
    valid_cluster = True
    for i in sorted(labels.unique()):
        submatrix = distance_matrix.loc[labels == i, labels == i]
        subm_max_closest_neighbor = submatrix.replace(0, np.inf).min().max()
        voter_count = info_df.loc[submatrix.index].shape[0]
        subm_max_distance = submatrix.max().max()
        if any([subm_max_closest_neighbor > closest_neighbor_lim,
                subm_max_distance > furthest_neighbor_lim,
                submatrix.shape[0] > max_addr,
                voter_count > max_voters
                ]):
            valid_cluster = False
            break
    return valid_cluster


def building_df_to_kml(df, voter_info, kml_path='kml_test.kml'):
    kml = simplekml.Kml()
    fol = kml.newfolder()
    for geoid, row in df.iterrows():
        pnt = fol.newpoint(name=row['label'],
                           coords=[(row['longitude'], row['latitude'])]
                           )
        pnt.description = str(row['cluster_id'])
        kml.save(kml_path)


def transform_to_printable(df):
    df = df.sort_index(level=['geo_id', 'Address Id'])
    printing_df = df.reset_index()[['VANID', 'Street Address', 'clean_name', 'cluster_id']]
    new_columns = ["Attempted", "Not Available", "Inaccessible",
                   "Ballot Received?",
                   "Completed Ballot Mailed?",
                   "Dropped at Dropbox?",
                   "Received by County?",
                   "Needs Help Completing?",
                   "Needs Ride?",
                   "Phone # for ride",
                   "Notes"]
    printing_df[new_columns] = ""
    printing_df.index = range(1, printing_df.shape[0] + 1)
    printing_df['Street'] = printing_df['Street Address'].str.split(' ').apply(
        lambda x: ' '.join(x[1:-1]))
    printing_df['HouseNumber'] = printing_df['Street Address'].str.split(' ').apply(
        lambda x: x[0])
    printing_df = printing_df.sort_values(by=['Street', 'HouseNumber']).drop(['Street', 'HouseNumber'], axis=1)
    return printing_df


@click.command()
@click.argument('fname', type=click.Path())
@click.option('--damping', default=0.7, help='Damping parameter for affinity propogation')
def run_pipe(fname, damping):
    prefix = pathlib.Path(fname).stem
    kml_fname = prefix + '.kml'
    latlon_fname = prefix + '_addresses_with_lat_long.csv'
    excel_fname = prefix + "_sheets.xlsx"
    stats_fname = prefix + "_stats.csv"


    voter_df = pd.read_csv(fname).drop_duplicates()
    voter_df['full_address'] = voter_df['Street Address'] + ', ' + voter_df['City'] + ' ' + \
                               voter_df['State/Province'] + ' ' + voter_df['Zip/Postal'].astype(str)
    voter_df = voter_df.set_index(INDEX_COLS).sort_index()

    # Add latitude and longitude columns
    voter_df['latitude'] = None
    voter_df['longitude'] = None

    if os.path.exists(latlon_fname):
        voter_df_with_latlon = pd.read_csv(latlon_fname).set_index(INDEX_COLS).sort_index()
    else:
        # Geocode each address
        for i, row in voter_df.iterrows():
            # lat, lng = get_lat_long(row['full_address'])
            addr, lat, lng = validate_address(row)
            voter_df.loc[i, 'Street Address'] = addr
            voter_df.loc[i, 'latitude'] = lat
            voter_df.loc[i, 'longitude'] = lng
            time.sleep(0.02)
        voter_df.to_csv(latlon_fname, index=True)
        voter_df_with_latlon = voter_df.copy()
        print(f"Geocoding complete. Saved to {latlon_fname}")

    voter_df_with_latlon['geo_id'] = floats_to_str(
        voter_df_with_latlon['latitude']) + floats_to_str(voter_df_with_latlon['longitude'])
    voter_df_with_latlon = voter_df_with_latlon.set_index('geo_id', append=True)
    voter_df_with_latlon['clean_name'] = clean_names(voter_df_with_latlon['Name'])

    # For clustering, we actually expect clustering to happen by the geo_id, but we use the address_id as a proxy because we expect all addresses within the same house to
    unique_addresses = voter_df_with_latlon.groupby(
        level=['Address Id']).first()  # Just get unique addresses

    # Outlier Handling
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    unique_addresses[['x', 'y']] = unique_addresses.apply(
        lambda row: transformer.transform(row['longitude'], row['latitude']), axis=1,
        result_type="expand")
    X = unique_addresses[['x', 'y']].copy()

    def identify_outliers(X, eps=0.25 * 5280 / 3.281, min_samples=5):
        cluster = DBSCAN(min_samples=5, eps=eps)
        labels = cluster.fit_predict(X)  # Offset by one to make them human-readable
        return labels < 0

    outliers = identify_outliers(X, eps=0.25 * 5280 / 3.281, min_samples=5)
    X.loc[:, 'outlier'] = outliers

    # Plotting outliers with map
    gdf = gpd.GeoDataFrame(X, geometry=gpd.points_from_xy(X['x'], X['y']), crs="EPSG:32618")
    ax = gdf.to_crs(epsg=3857).plot(figsize=(10, 10),
                                    column=gdf['outlier'].astype(int),
                                    cmap='bwr',
                                    marker='o', )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.title(f'Region "{prefix}" outliers')
    plt.axis('off')
    plt.savefig(f'{prefix}_outliers.png')

    unique_addresses = unique_addresses.loc[X[~ X['outlier']].index]
    voter_df_with_latlon = voter_df_with_latlon.loc[unique_addresses.index]

    distance_matrix = get_distance_matrix_for_addresses(unique_addresses)

    building_latlon = voter_df_with_latlon[['latitude', 'longitude']].groupby(
        level='geo_id').first()
    building_voter_count = voter_df_with_latlon['Street Address'].groupby(
        level='geo_id').count().rename('voter_count')
    building_addr = voter_df_with_latlon.sort_index(level='geo_id').groupby(level='geo_id').apply(
        lambda x: longest_common_word_prefix(x['Street Address'].tolist()).strip())
    building_addr.name = "building_address"

    building_table = building_latlon.join(building_voter_count).join(building_addr)
    building_table['label'] = building_table['building_address'] + ': ' + building_table[
        'voter_count'].apply(lambda x: f'{x} voter' if x == 1 else f'{x} voters')

    X = unique_addresses[['x', 'y']].copy()

    title = f'Affinity propogation, damping={damping:.2f}'
    cluster = AffinityPropagation(damping=0.7)
    model_input = X

    labels = cluster.fit_predict(model_input) + 1  # Offset by one to make them human-readable
    X.loc[:, 'cluster_id'] = labels
    addr_labels = X['cluster_id']

    # Plot cluster key with map
    gdf = gpd.GeoDataFrame(X, geometry=gpd.points_from_xy(X['x'], X['y']), crs="EPSG:32618")
    gdf = gdf.to_crs(epsg=3857)
    ax = gdf.plot(figsize=(10, 10), column=gdf['cluster_id'].astype(int), cmap='nipy_spectral',
              marker='o')
    for t, s in gdf.groupby('cluster_id').geometry.apply(lambda x: x.union_all().centroid).items():
        plt.text(s.x, s.y, str(t), fontsize=16,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.8)
                 )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.axis('off')
    plt.title(f'Region "{prefix}" {title}')
    plt.tight_layout()
    plt.savefig(f'{prefix}_cluster_key.png', dpi=150)

    # Create statistics on cluster size
    stats = cluster_stats(addr_labels, distance_matrix, voter_df_with_latlon).T
    stats.index.name = 'cluster_id'
    stats[['max_closest_neighbor', 'max_span']] = stats[
        ['max_closest_neighbor', 'max_span']].round()
    stats = stats.rename({'max_closest_neighbor': 'Max distance between neighbors (ft)',
                          'max_span': 'Distance to cross turf (ft)', }, axis=1)
    stats['prefix'] = prefix
    stats = stats.set_index('prefix', append=True)
    stats = stats.swaplevel('prefix', 'cluster_id')
    stats.to_csv(stats_fname)

    voter_df_with_latlon = voter_df_with_latlon.join(addr_labels.copy(), how='left')

    count_clusters_at_geoid = voter_df_with_latlon['cluster_id'].groupby(level='geo_id').apply(lambda x: len(x.unique()))
    geoids_with_multiple_clusters = count_clusters_at_geoid[count_clusters_at_geoid > 1].rename('num_clusters')
    if len(geoids_with_multiple_clusters) > 0:
        print("Clusters split within buildings! Geoids with multiple clusters:")
        print(pd.DataFrame(geoids_with_multiple_clusters))
        rows_to_drop = pd.Series(voter_df_with_latlon.index.get_level_values('geo_id')).isin(
            geoids_with_multiple_clusters.index).values
        print(f"Dropping a total of {rows_to_drop.sum()} rows due to geoid clashes")
        voter_df_with_latlon = voter_df_with_latlon[~rows_to_drop]

    building_table = building_table.join(
        voter_df_with_latlon['cluster_id'].groupby(level='geo_id').first())

    # Output creation
    building_df_to_kml(building_table, voter_df_with_latlon, kml_fname)

    with pd.ExcelWriter(excel_fname) as writer:
        # Group by 'PopulationID' and write each group to a separate sheet
        for cluster, group in voter_df_with_latlon.groupby('cluster_id'):
            printable_df = transform_to_printable(group)
            # Write each group to a different sheet named after the PopulationID
            printable_df.index = range(printable_df.shape[0])
            printable_df.drop('cluster_id', axis=1).to_excel(writer, sheet_name=f'List {cluster}',
                                                             index=True)


if __name__ == "__main__":
    run_pipe()
