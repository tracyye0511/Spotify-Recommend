# 1. Load Dataset
import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# -spotify-tracks-dataset: Dataset lagu secara general
tracks_df = pd.read_csv('1dataset.csv')

# dataset-of-songs-in-spotify: Dataset lagu secara general
genres_v2_df = pd.read_csv(
    'genres_v2.csv',
    dtype={19: str} # Karena pas proses sempat type error
)

# spotifyclassification: User profile 1 (di Kaggle, dataset nya berjudul Spotify Song Attributes)
user_profile_1_df = pd.read_csv('data4.csv')

# spotify-recommendation: User profile 2
user_profile_2_data_df = pd.read_csv('data3.csv')

# Print null counts HANYA untuk kolom dengan missing values (SEBELUM removal)
def print_null_counts(df, df_name):
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print(f"Missing values in {df_name}:\n{null_counts}\n")
    else:
        print(f"No missing values found in {df_name}\n")

print_null_counts(tracks_df, "tracks_df")
print_null_counts(genres_v2_df, "genres_v2_df")
print_null_counts(user_profile_1_df, "user_profile_1_df")
print_null_counts(user_profile_2_data_df, "user_profile_2_data_df")

# 2. Preprocessing Dataset

# Fitur numerik untuk digunakan di semua dataset
numerical_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature'
]

# Print dataset shapes original
print("\nOriginal dataset shapes:")
print(f"tracks_df: {tracks_df.shape}")
print(f"genres_v2_df: {genres_v2_df.shape}")
print(f"user_profile_1_df: {user_profile_1_df.shape}")
print(f"user_profile_2_data_df: {user_profile_2_data_df.shape}")

# tracks_df - hanya simpan fitur numerik
tracks_features = tracks_df.drop(columns=[
    col for col in tracks_df.columns
    if col not in numerical_features
], errors='ignore')

# genres_v2_df - hanya simpan fitur numerik
genres_v2_features = genres_v2_df.drop(columns=[
    col for col in genres_v2_df.columns
    if col not in numerical_features
], errors='ignore')

# Untuk user_profile_1_df, ubah nama 'target' ke 'liked' supaya nama kolom sama/standar
if 'target' in user_profile_1_df.columns and 'liked' not in user_profile_1_df.columns:
    user_profile_1_df = user_profile_1_df.rename(columns={'target': 'liked'})

# Ambil kolom fitur dan target (prediksi) dari tiap user profile
user_profile_1_features = user_profile_1_df[numerical_features + ['liked']]
user_profile_2_features = user_profile_2_data_df[numerical_features + ['liked']]

# Print dataset shapes setelah preprocessing
print("\nProcessed dataset shapes:")
print(f"tracks_features: {tracks_features.shape}")
print(f"genres_v2_features: {genres_v2_features.shape}")
print(f"user_profile_1_features: {user_profile_1_features.shape}")
print(f"user_profile_2_features: {user_profile_2_features.shape}")

# Print jumlah null setelah preprocessing
print("\nNull counts after preprocessing:")
print_null_counts(tracks_features, "tracks_features")
print_null_counts(genres_v2_features, "genres_v2_features")
print_null_counts(user_profile_1_features, "user_profile_1_features")
print_null_counts(user_profile_2_features, "user_profile_2_features")

# Print distribusi user profile (0 = tidak suka, 1 = suka)
print("\nUser Profile 1 'liked' distribution:")
print(user_profile_1_features['liked'].value_counts())

print("\nUser Profile 2 'liked' distribution:")
print(user_profile_2_features['liked'].value_counts())

# 3. Feature Engineering

# Untuk dataset general, digabungkan menjadi satu dataframe dengan pd.concat untuk mempermudah
all_song_data = pd.concat([tracks_features, genres_v2_features], ignore_index=True)

# Buat scaler dan fit scaler tersebut berdasarkan all_song_data
scaler = StandardScaler()
scaler.fit(all_song_data[numerical_features])

# Gunakan scaler tadi untuk mengubah fitur numerik berdasarkan hasil scaling
scaled_song_data = scaler.transform(all_song_data[numerical_features])
all_songs = pd.DataFrame(scaled_song_data, columns=numerical_features)

print(f"\nCombined song catalog shape: {all_songs.shape}")

# 4. Model Training and Evaluation

# Constants
WEIGHT_FACTOR = 0.3  # Faktor bobot randomness dalam rekomendasi (untuk menhindari overfitting)
RANDOM_STATE = 42  # Supaya hasil pelatihan model bisa di akses ulang (hasil konsisten)
TOP_N = 10  # Jumlah lagu yang direkomendasi


def content_based_recommendation(user_profile, song_catalog, top_n=TOP_N,
                                 initial_pool_size=50, random_state=RANDOM_STATE,
                                 weight_factor=WEIGHT_FACTOR):
    # Set random_state
    random.seed(random_state)
    np.random.seed(random_state)

    # Hitung vektor user profile dari lagu yang disukai
    liked_songs = user_profile[user_profile['liked'] == 1][numerical_features]
    user_profile_vector = liked_songs.mean().values.reshape(1, -1)

    # Hitung cosine similarity antara user profile dan semua lagu (yang ada di all_song_data)
    song_vectors = song_catalog[numerical_features].values
    similarities = cosine_similarity(user_profile_vector, song_vectors)[0]

    # Susun lagu dari song_catalog berdasarkan hasil cosine similarity
    top_indices = np.argsort(similarities)[::-1][:initial_pool_size]
    initial_pool = song_catalog.iloc[top_indices]

    # Kemungkinan lagu terpilih (SEBELUM pengacakan)
    probabilities = similarities[top_indices] / np.sum(similarities[top_indices])

    # Bobot randomness supaya model tidak selalu rekomendasi lagu sama
    random_weights = np.random.rand(initial_pool_size)
    random_weights /= np.sum(random_weights)

    #  Kemungkinan lagu terpilih baru berdasarkan bobot randomness (SESUDAH PENGACAKAN)
    adjusted_probabilities = (1 - weight_factor) * probabilities + weight_factor * random_weights

    # Pilih rekomendasi akhir berdasarkan semua variabel tadi
    final_top_indices = np.random.choice(
        initial_pool.index,
        size=min(top_n, len(initial_pool)),
        replace=False,
        p=adjusted_probabilities
    )

    recommendations = song_catalog.iloc[final_top_indices]
    confidence_scores = similarities[final_top_indices]

    return recommendations, confidence_scores


def calculate_recommendation_diversity(recommendations):
    if len(recommendations) <= 1:
        return 1.0  # Bila hanya ada satu song yang direkomendasi, maka nilai diversity maksimal (1.0)

    # Hitung pairwise similarities antara semua rekomendasi
    song_vectors = recommendations[numerical_features].values
    pairwise_similarities = cosine_similarity(song_vectors)

    # Supaya lagu yang sama tidak saling dibandingkan
    avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])

    # Diversity adalah inverse dari similarity
    diversity = 1 - avg_similarity

    return diversity


def evaluate_recommender(user_profile, song_catalog, test_size=0.2, random_state=RANDOM_STATE):
    print(f"Evaluating with {len(user_profile)} songs ({user_profile['liked'].sum()} liked)")

    # Train-test split
    train_df, test_df = train_test_split(
        user_profile,
        test_size=test_size,
        stratify=user_profile['liked'],
        random_state=random_state
    )

    print(f"Training: {len(train_df)} songs ({train_df['liked'].sum()} liked)")
    print(f"Testing: {len(test_df)} songs ({test_df['liked'].sum()} liked)")

    # Buat user profile dari training data
    train_liked = train_df[train_df['liked'] == 1]
    user_profile_vector = train_liked[numerical_features].mean().values.reshape(1, -1)

    # Hitung cosine similarity untuk test songs
    test_vectors = test_df[numerical_features].values
    similarities = cosine_similarity(user_profile_vector, test_vectors)[0]

    # Evaluasi performa model berdasarkan keyakinan prediksi
    results = []
    for percentile in range(50, 100, 5):
        threshold = np.percentile(similarities, percentile)
        predictions = (similarities >= threshold).astype(int)

        # Hitung metrik/nilai evaluasi
        precision = precision_score(test_df['liked'].values, predictions)
        recall = recall_score(test_df['liked'].values, predictions)
        f1 = f1_score(test_df['liked'].values, predictions)

        results.append({
            'threshold_percentile': 100 - percentile,
            'threshold_value': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_likes': sum(predictions),
            'actual_likes': sum(test_df['liked'])
        })

    # Cari hasil terbaik berdasarkan f1 score
    best_result = max(results, key=lambda x: x['f1'])
    return best_result, results


# Jalankan rekomendasi dasar untuk setiap pengguna untuk memverifikasi fungsionalitas
print("\nGenerating recommendations for User 1:")
recommendations_1, confidence_scores_1 = content_based_recommendation(
    user_profile_1_features, all_songs)
diversity_1 = calculate_recommendation_diversity(recommendations_1)
print(f"Generated {len(recommendations_1)} recommendations with diversity: {diversity_1:.4f}")

print("\nGenerating recommendations for User 2:")
recommendations_2, confidence_scores_2 = content_based_recommendation(
    user_profile_2_features, all_songs)
diversity_2 = calculate_recommendation_diversity(recommendations_2)
print(f"Generated {len(recommendations_2)} recommendations with diversity: {diversity_2:.4f}")

# Mengevaluasi performa rekomendasi model
print("\n--- METRICS BASED ON HISTORICAL USER DATA ---")
print("\nUser 1 Evaluation:")
user1_best, user1_all_results = evaluate_recommender(user_profile_1_features, all_songs)
print(f"Best F1 Score: {user1_best['f1']:.4f} (at threshold for top {user1_best['threshold_percentile']}%)")
print(f"Precision: {user1_best['precision']:.4f}, Recall: {user1_best['recall']:.4f}")

print("\nUser 2 Evaluation:")
user2_best, user2_all_results = evaluate_recommender(user_profile_2_features, all_songs)
print(f"Best F1 Score: {user2_best['f1']:.4f} (at threshold for top {user2_best['threshold_percentile']}%)")
print(f"Precision: {user2_best['precision']:.4f}, Recall: {user2_best['recall']:.4f}")