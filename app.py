from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
from tensorflow.keras.models import load_model
from math import radians
import os

# Inisialisasi Flask
app = Flask(__name__)

# Load dataset
file_path = "Dataset Model.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)
dataframes = []

for sheet_name, sheet_df in sheets.items():
    sheet_df['Kota'] = sheet_name
    dataframes.append(sheet_df)

df = pd.concat(dataframes, ignore_index=True)

# Normalize Latitude and Longitude columns
scaler = MinMaxScaler()
pd_normalized = scaler.fit_transform(df[['Latitude', 'Longitude']])
df_normalized = pd.DataFrame(pd_normalized)

# Load pretrained autoencoder and encoder
model_path = "autoencoder_model.h5"
if os.path.exists(model_path):
    autoencoder = load_model(model_path)
    encoder = load_model(model_path, compile=False)  # Menggunakan encoder langsung
else:
    raise FileNotFoundError("Model autoencoder tidak ditemukan.")

# Cluster embeddings
embeddings = encoder.predict(pd_normalized.astype(np.float32))  # Pastikan float32
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(embeddings)

# Function to find the closest locations
def find_nearest_locations_with_rating(user_location, df, kmeans, encoder, scaler, n_neighbors, weight_distance, weight_rating):
    user_location_arr = np.array([user_location], dtype=np.float32)  # Konversi ke float32
    user_location_normalized = scaler.transform(user_location_arr).astype(np.float32)
    user_location_embedding = encoder.predict(user_location_normalized)
    new_cluster = kmeans.predict(user_location_embedding)[0]

    def prepare_coordinates(lat, lon):
        return np.array([[radians(lat), radians(lon)]])
    
    user_loc_radians = prepare_coordinates(user_location[0], user_location[1])

    cluster_radius = 1
    nearby_clusters = np.where(
        euclidean_distances(kmeans.cluster_centers_[new_cluster].reshape(1, -1),
                            kmeans.cluster_centers_) < cluster_radius)[1]

    potential_locations = df[df['Cluster'].isin(nearby_clusters)].copy()
    if len(potential_locations) == 0:
        potential_locations = df[df['Cluster'] == new_cluster].copy()

    locations_radians = np.radians(
        potential_locations[['Latitude', 'Longitude']].values
    )
    distances = haversine_distances(user_loc_radians, locations_radians)[0] * 6371

    potential_locations['Jarak_km'] = distances
    potential_locations['Score'] = (
        weight_distance * potential_locations['Jarak_km'] +
        weight_rating * (-potential_locations['Rating'])
    )
    nearest_locations = potential_locations.nsmallest(n_neighbors, 'Score')

    result = nearest_locations.copy()
    result['Jarak_km'] = result['Jarak_km'].round(2)
    return result

# Endpoint rekomendasi lokasi
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Ambil data dari request
        data = request.json
        latitude = float(data['latitude'])  # Konversi ke float
        longitude = float(data['longitude'])  # Konversi ke float

        # Cari lokasi terdekat
        place_recommendation = find_nearest_locations_with_rating(
            user_location=[latitude, longitude],
            df=df,
            kmeans=kmeans,
            encoder=encoder,
            scaler=scaler,
            n_neighbors=5,
            weight_distance=0.5,
            weight_rating=0.5
        )

        # Format hasil dalam bentuk JSON
        recommendation_json = place_recommendation[['Nama Tempat', 'Latitude', 'Longitude', 'Jarak_km', 'Rating']].to_dict(orient='records')
        return jsonify(recommendation_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
