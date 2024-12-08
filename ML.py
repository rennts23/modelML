import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
from math import radians
import json
import os

import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Save the model to disk
tf.saved_model.save(model, 'saved_model/')

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

# Define the autoencoder model
input_dim = df_normalized.shape[1]  # Number of features (2: lat, lon)
encoding_dim = 2  # Latent space dimension

def create_autoencoder():
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    return autoencoder, encoder

model_path = "autoencoder_model.h5"
if os.path.exists(model_path):
    # Load pre-trained model
    autoencoder = load_model(model_path)
    encoder = Model(autoencoder.input, autoencoder.layers[-3].output)  # Extract encoder
    print("Model loaded from file.")
else:
    # Train new model
    autoencoder, encoder = create_autoencoder()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.fit(
        df_normalized, df_normalized,
        epochs=100, batch_size=32, verbose=1, validation_split=0.1
    )
    autoencoder.save(model_path)
    print("Model trained and saved.")

# Get the embedding (compression representation) of the data
embeddings = encoder.predict(pd_normalized)
embedding_df = pd.DataFrame(embeddings, columns=['dim1', 'dim2'])

# Cluster embeddings
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(embeddings)

# Function to find the closest locations
def find_nearest_locations_with_rating(user_location, df, kmeans, encoder, scaler, n_neighbors, weight_distance, weight_rating):
    user_location_arr = np.array([user_location])
    user_location_normalized = scaler.transform(user_location_arr)
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

# Load user input from JSON
input_file = "user_input.json"
output_file = "place_recommendation.json"

with open(input_file, 'r') as f:
    user_data = json.load(f)
user_location = user_data['location']

# Find recommended service places
place_recommendation = find_nearest_locations_with_rating(
    user_location=user_location,
    df=df,
    kmeans=kmeans,
    encoder=encoder,
    scaler=scaler,
    n_neighbors=5,
    weight_distance=0.5,
    weight_rating=0.5
)

# Convert recommendations to JSON and save
recommendation_json = place_recommendation[['Nama Tempat', 'Latitude', 'Longitude', 'Jarak_km', 'Rating']].to_json(orient='records')
with open(output_file, 'w') as f:
    f.write(recommendation_json)

print("Rekomendasi tempat servis yang ditemukan telah disimpan ke JSON.")
print(recommendation_json)
