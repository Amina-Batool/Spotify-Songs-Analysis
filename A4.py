import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import json
import logging

# Set up logging for debugging
logging.basicConfig(filename='A4_debug.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

# ====================
# Data Loading and Preprocessing
# ====================
def load_and_preprocess_data(filepath):
    print("[CHECKPOINT] Starting Data Loading...")
    try:
        df = pd.read_excel(filepath, sheet_name=0)
    except Exception as e:
        logging.error(f"Failed to load Excel file: {str(e)}")
        raise ValueError(f"Error loading {filepath}: {str(e)}")
    
    # Log raw dataset details
    logging.debug(f"Raw dataset columns: {list(df.columns)}")
    logging.debug(f"Raw dataset dtypes:\n{df.dtypes}")
    logging.debug(f"Raw snapshot_date sample: {df['snapshot_date'].head().to_list()}")
    logging.debug(f"Raw album_release_date sample: {df['album_release_date'].head().to_list()}")
    
    # Initialize LabelEncoder for genre
    le_genre = LabelEncoder()
    
    # Check for required columns
    required_columns = ['snapshot_date', 'album_release_date', 'popularity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print("[CHECKPOINT] Starting Data Preprocessing...")
    # Convert dates to datetime with Excel origin (for Excel serial dates)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], origin='1899-12-30', unit='D')
    df['album_release_date'] = pd.to_datetime(df['album_release_date'], origin='1899-12-30', unit='D')
    
    # Log converted dates and invalid counts
    logging.debug(f"Converted snapshot_date sample: {df['snapshot_date'].head().to_list()}")
    logging.debug(f"Converted album_release_date sample: {df['album_release_date'].head().to_list()}")
    logging.debug(f"Invalid snapshot_date count: {df['snapshot_date'].isna().sum()}")
    logging.debug(f"Invalid album_release_date count: {df['album_release_date'].isna().sum()}")
    
    # Calculate days_since_release
    df['days_since_release'] = (df['snapshot_date'] - df['album_release_date']).dt.days
    df['days_since_release'] = df['days_since_release'].clip(lower=0)
    
    # Validate input data
    df['duration_ms'] = df['duration_ms'].clip(lower=0)
    
    # Convert categorical/binary features
    df['is_explicit'] = df['is_explicit'].astype(int)
    
    # Audio-Specific Feature Engineering
    df['mood_score'] = (df['valence'] + df['energy'] + df['danceability']) / 3 * 100
    df['focus_metric'] = df['instrumentalness'] * (1 - df['speechiness'])
    df['duration_category'] = pd.cut(df['duration_ms'] / 60000, 
                                    bins=[0, 3, 5, np.inf], 
                                    labels=['Short', 'Medium', 'Long'])
    
    # Time-Based Feature Engineering
    df['age_category'] = pd.cut(df['days_since_release'],
                                bins=[0, 30, 365*5, np.inf],
                                labels=['New', 'Recent', 'Old'])
    
    # Derive season_released based on album_release_date month
    df['month_released'] = df['album_release_date'].dt.month
    df['season_released'] = df['month_released'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall')
    
    logging.debug(f"Season distribution:\n{df['season_released'].value_counts().to_dict()}")
    
    # Derive genre if not present
    if 'genre' not in df.columns:
        logging.debug("Genre column not found. Deriving genre based on audio features...")
        def assign_genre(row):
            if row['instrumentalness'] > 0.7 and row['speechiness'] < 0.2:
                return 'Classical/Instrumental'
            elif row['danceability'] > 0.7 and row['energy'] > 0.7 and row['tempo'] > 120:
                return 'Electronic/Dance'
            elif row['speechiness'] > 0.3 and row['danceability'] > 0.6:
                return 'Hip-Hop/Rap'
            elif row['loudness'] > -10 and row['energy'] > 0.6 and row['danceability'] < 0.6:
                return 'Rock'
            else:
                return 'Pop'
        df['genre'] = df.apply(assign_genre, axis=1)
    else:
        genre_mapping = {
            'pop': 'Pop', 'dance pop': 'Pop', 'pop rock': 'Pop',
            'rock': 'Rock', 'hard rock': 'Rock', 'metal': 'Rock', 'punk': 'Rock', 'alternative rock': 'Rock',
            'hip hop': 'Hip-Hop/Rap', 'rap': 'Hip-Hop/Rap', 'trap': 'Hip-Hop/Rap', 'r&b': 'Hip-Hop/Rap',
            'classical': 'Classical/Instrumental', 'instrumental': 'Classical/Instrumental', 'orchestral': 'Classical/Instrumental',
            'electronic': 'Electronic/Dance', 'edm': 'Electronic/Dance', 'dance': 'Electronic/Dance', 'house': 'Electronic/Dance', 'techno': 'Electronic/Dance'
        }
        df['genre'] = df['genre'].str.lower().map(genre_mapping).fillna('Pop')
    logging.debug(f"Genre distribution:\n{df['genre'].value_counts().to_dict()}")
    
    # Features to drop for model training
    drop_columns = ['spotify_id', 'name', 'artists', 'album_name', 
                    'snapshot_date', 'album_release_date', 'country', 
                    'daily_rank', 'daily_movement', 'weekly_movement',
                    'month_released']
    
    # Keep a copy of the full dataset for the frontend
    full_df = df.copy()
    
    # Drop columns for training
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    
    # Handle missing values for numeric columns
    df = df.replace(['NaN', 'missing', ''], np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Log column types before categorical handling
    logging.debug(f"Column types before categorical handling:\n{df.dtypes}")
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    logging.debug(f"Categorical columns detected: {list(categorical_cols)}")
    
    for col in categorical_cols:
        logging.debug(f"Unique values in {col}: {df[col].unique()}")
        # Convert category to string and handle missing values
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(str)
        if df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    # Explicitly encode categorical columns
    print("[CHECKPOINT] Encoding Categorical Features...")
    le_duration = LabelEncoder()
    le_age = LabelEncoder()
    
    if 'duration_category' in df.columns:
        df['duration_category'] = df['duration_category'].astype(str)  # Ensure string type
        df['duration_category'] = le_duration.fit_transform(df['duration_category'])
        logging.debug(f"Encoded duration_category: {df['duration_category'].unique()}")
    
    if 'age_category' in df.columns:
        df['age_category'] = df['age_category'].astype(str)  # Ensure string type
        df['age_category'] = le_age.fit_transform(df['age_category'])
        logging.debug(f"Encoded age_category: {df['age_category'].unique()}")
    
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype(str)  # Ensure string type
        df['genre'] = le_genre.fit_transform(df['genre'])
        logging.debug(f"Encoded genre: {df['genre'].unique()}, Classes: {le_genre.classes_}")
    
    if 'season_released' in df.columns:
        # One-hot encode season_released
        df = pd.get_dummies(df, columns=['season_released'], prefix='season', dummy_na=False)
        for col in ['season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']:
            if col in df.columns:
                df[col] = df[col].astype(int)
        logging.debug(f"Columns after one-hot encoding season_released: {list(df.columns)}")
    
    # Drop any remaining categorical or non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logging.warning(f"Dropping unexpected non-numeric columns: {list(non_numeric_cols)}")
        df = df.drop(columns=non_numeric_cols)
    
    # Final check for non-numeric columns
    logging.debug(f"Column types after encoding and cleanup:\n{df.dtypes}")
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logging.error(f"Non-numeric columns after final cleanup: {list(non_numeric_cols)}")
        raise ValueError("Non-numeric columns remain after preprocessing.")
    
    print("[CHECKPOINT] Data Preprocessing Completed.")
    
    # Save audio feature statistics for frontend
    audio_stats = {
        'mood_score': {
            'mean': float(df['mood_score'].mean()),
            'std': float(df['mood_score'].std()),
            'min': float(df['mood_score'].min()),
            'max': float(df['mood_score'].max())
        },
        'focus_metric': {
            'mean': float(df['focus_metric'].mean()),
            'std': float(df['focus_metric'].std()),
            'min': float(df['focus_metric'].min()),
            'max': float(df['focus_metric'].max())
        },
        'duration_category': df['duration_category'].value_counts().to_dict(),
        'genre': full_df['genre'].value_counts().to_dict()
    }
    with open("audio_features.json", "w") as f:
        json.dump(audio_stats, f)
    print("[CHECKPOINT] Audio Feature Statistics Saved...")
    
    # Save time-based trend statistics for frontend
    time_trends = {
        'days_since_release': {
            'mean': float(df['days_since_release'].mean()),
            'std': float(df['days_since_release'].std()),
            'min': float(df['days_since_release'].min()),
            'max': float(df['days_since_release'].max())
        },
        'age_category': df['age_category'].value_counts().to_dict(),
        'season_released': full_df['season_released'].value_counts().to_dict()
    }
    with open("time_trends.json", "w") as f:
        json.dump(time_trends, f)
    print("[CHECKPOINT] Time-Based Trends Saved...")
    
    return df, full_df, le_genre

# ====================
# Model Evaluation Functions
# ====================
def evaluate_model(model, name, X_test, y_test, le_genre):
    print(f"[CHECKPOINT] Generating Confusion Matrix for {name}...")
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    logging.debug(f"{name} Confusion Matrix:\n{cm}")
    logging.debug(f"{name} Prediction Distribution: {np.bincount(y_pred)}")
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # Check for repetitive scores
    metric_values = list(metrics.values())
    if len(set(metric_values)) < len(metric_values):
        logging.warning(f"{name} has repetitive metric values: {metrics}")
    
    # Generate detailed classification report
    report = classification_report(y_test, y_pred, target_names=le_genre.classes_, output_dict=True)
    with open(f"{name}_classification_report.json", "w") as f:
        json.dump(report, f)
    logging.debug(f"{name} Classification Report:\n{report}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_genre.classes_, yticklabels=le_genre.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()
    
    return metrics

def plot_feature_importance(model, feature_names):
    print("[CHECKPOINT] Generating Feature Importance Plot...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Important Features")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.savefig("feature_importance.png")
        plt.close()

# ====================
# Explainable AI Functions
# ====================
def create_surrogate_model(model, X_test, y_test_pred, feature_names, max_depth=3):
    print("[CHECKPOINT] Creating Surrogate Model...")
    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    surrogate.fit(X_test, y_test_pred)
    
    rules = export_text(surrogate, feature_names=list(feature_names))
    with open("surrogate_rules.txt", "w") as f:
        f.write(rules)
    print("[CHECKPOINT] Surrogate Model Rules Saved...")
    
    return surrogate
def generate_counterfactuals(model, scaler, X, feature_names, le_genre, target_class=1, n_features=3, max_attempts=100):
    print("[CHECKPOINT] Generating Counterfactual Explanations...")
    
    # Define continuous features to perturb
    continuous_features = [
        f for f in feature_names
        if f not in ['duration_category', 'age_category', 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter', 'genre']
    ]
    if len(continuous_features) == 0:
        logging.error("No continuous features available for counterfactual generation.")
        comparison = pd.DataFrame({
            'Feature': feature_names,
            'Original': [0] * len(feature_names),
            'Counterfactual': [0] * len(feature_names),
            'Change': [0] * len(feature_names),
            'Status': ['No continuous features'] * len(feature_names)
        })
        comparison.to_csv("counterfactual_explanation.csv", index=False)
        print("[CHECKPOINT] Counterfactual Explanations Saved (Empty due to no continuous features)...")
        return comparison
    
    # Select a random sample that is not in the target class
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    non_target_indices = np.where(predictions != target_class)[0]
    if len(non_target_indices) == 0:
        logging.warning("All samples are already in the target class. Using a random sample.")
        sample_idx = np.random.randint(0, len(X))
    else:
        sample_idx = np.random.choice(non_target_indices)
    
    original_features = X.iloc[sample_idx].copy()
    original_df = pd.DataFrame([original_features], columns=feature_names)
    scaled_sample = scaler.transform(original_df)
    original_pred = model.predict(scaled_sample)[0]
    
    logging.debug(f"Selected sample index: {sample_idx}, Original prediction: {le_genre.classes_[original_pred]}")
    
    # Initialize counterfactual as the original sample
    counterfactual_df = original_df.copy()
    
    # Select top N continuous features based on importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        selected_features = feature_importance[continuous_features].nlargest(n_features).index.tolist()
    else:
        selected_features = continuous_features[:n_features]
    
    logging.debug(f"Selected features for perturbation: {selected_features}")
    
    for feature in selected_features:
        logging.debug(f"Feature {feature}: min={X[feature].min()}, max={X[feature].max()}, std={X[feature].std()}")
    
    # Perturb features to find a counterfactual
    for attempt in range(max_attempts):
        temp_df = counterfactual_df.copy()
        perturbations = {}
        for feature in selected_features:
            std = X[feature].std() or 1
            perturbation = np.random.uniform(-0.5, 0.5) * std
            perturbations[feature] = perturbation
            temp_df[feature] += perturbation
            temp_df[feature] = temp_df[feature].clip(X[feature].min(), X[feature].max())
        
        logging.debug(f"Attempt {attempt + 1}: Perturbations={perturbations}")
        
        scaled_temp = scaler.transform(temp_df)
        current_pred = model.predict(scaled_temp)[0]
        
        if current_pred == target_class:
            counterfactual_df = temp_df
            logging.debug(f"Counterfactual found after {attempt + 1} attempts")
            break
    else:
        logging.warning("No counterfactual found within max attempts")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'Original': original_features.values,
        'Counterfactual': counterfactual_df.iloc[0].values,
        'Change': counterfactual_df.iloc[0].values - original_features.values,
        'Status': ['Success' if current_pred == target_class else 'No counterfactual found'] * len(feature_names)
    })
    
    comparison.to_csv("counterfactual_explanation.csv", index=False)
    print("[CHECKPOINT] Counterfactual Explanations Saved...")
    return comparison

# ====================
# Clustering Analysis
# ====================
def perform_clustering(X_scaled, y, full_df):
    print("[CHECKPOINT] Starting Clustering Analysis...")
    silhouette_scores = []
    for k in range(3, 6):  # Reduced range for faster execution
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    optimal_k = np.argmax(silhouette_scores) + 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    metrics = {
        "Optimal Clusters": int(optimal_k),
        "Silhouette Score": silhouette_score(X_scaled, clusters),
        "Davies-Bouldin Score": davies_bouldin_score(X_scaled, clusters),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, clusters)
    }
    
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)
    representative_indices = []
    for i in range(optimal_k):
        cluster_center = kmeans.cluster_centers_[i]
        _, indices = nn.kneighbors([cluster_center])
        representative_indices.extend(indices[0])
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print("[CHECKPOINT] Generating Clustering Visualization...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(X_pca[representative_indices, 0], X_pca[representative_indices, 1], 
                s=100, edgecolors='red', facecolors='none', label='Representative Songs')
    plt.title("K-Means Clusters with Representative Songs")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.where(outliers == -1, 'red', 'blue'), alpha=0.6)
    plt.title("Anomaly Detection (Red = Outliers)")
    
    plt.tight_layout()
    plt.savefig("advanced_clustering_analysis.png")
    plt.close()
    
    # Save anomaly detection statistics for frontend
    outlier_df = full_df[outliers == -1]
    anomaly_stats = {
        'num_outliers': int(len(outlier_df)),
        'outlier_characteristics': {
            'mean_popularity': float(outlier_df['popularity'].mean()) if len(outlier_df) > 0 else 0,
            'mean_energy': float(outlier_df['energy'].mean()) if len(outlier_df) > 0 else 0,
            'mean_danceability': float(outlier_df['danceability'].mean()) if len(outlier_df) > 0 else 0
        },
        'sample_outliers': outlier_df[['name', 'artists', 'popularity']].head(5).to_dict(orient='records')
    }
    with open("anomaly_detection.json", "w") as f:
        json.dump(anomaly_stats, f)
    print("[CHECKPOINT] Anomaly Detection Statistics Saved...")
    
    return metrics, clusters, representative_indices, outliers

# ====================
# Visualization Helpers
# ====================
def plot_seasonal_trends(df):
    print("[CHECKPOINT] Generating Seasonal Trends Plot...")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='season_released', y='popularity', hue='genre', data=df)
    plt.title("Song Popularity by Release Season and Genre")
    plt.savefig("seasonal_trends.png")
    plt.close()

def plot_algorithm_comparison(knn_metrics, nb_metrics, rf_metrics):
    print("[CHECKPOINT] Generating Algorithm Comparison Plot...")
    metrics_df = pd.DataFrame({
        "KNN": knn_metrics,
        "Naïve Bayes": nb_metrics,
        "Random Forest": rf_metrics
    }).T
    
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.close()

# ====================
# Main Execution
# ====================
def main():
    print("[CHECKPOINT] Starting Main Execution...")
    df, full_df, le_genre = load_and_preprocess_data('spotify_songs_dataset.xlsx')
    
    target_column = 'genre'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logging.debug(f"Feature types before scaling:\n{X.dtypes}")
    
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logging.error(f"Non-numeric columns: {list(non_numeric_cols)}")
        raise ValueError("Non-numeric columns detected after encoding. Check preprocessing.")
    
    print("[CHECKPOINT] Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[CHECKPOINT] Splitting Data into Training and Testing Sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("[CHECKPOINT] Starting Model Training...")
    
    logging.debug("Tuning KNN...")
    param_grid_knn = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance']
    }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3)
    grid_knn.fit(X_train, y_train)
    knn = grid_knn.best_estimator_
    logging.debug(f"Best KNN params: {grid_knn.best_params_}")
    
    logging.debug("Tuning Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'class_weight': ['balanced', None]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3)
    grid_rf.fit(X_train, y_train)
    rf = grid_rf.best_estimator_
    logging.debug(f"Best RF params: {grid_rf.best_params_}")
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    print("[CHECKPOINT] Evaluating Models...")
    knn_metrics = evaluate_model(knn, "KNN", X_test, y_test, le_genre)
    nb_metrics = evaluate_model(nb, "Naïve Bayes", X_test, y_test, le_genre)
    rf_metrics = evaluate_model(rf, "Random Forest", X_test, y_test, le_genre)
    
    plot_feature_importance(rf, X.columns)
    
    y_test_pred = rf.predict(X_test)
    surrogate = create_surrogate_model(rf, X_test, y_test_pred, X.columns)
    
    target_class = le_genre.transform(['Pop'])[0]
    counterfactual = generate_counterfactuals(rf, scaler, X, X.columns, le_genre, target_class=target_class)
    
    plot_algorithm_comparison(knn_metrics, nb_metrics, rf_metrics)
    
    print("[CHECKPOINT] Performing Clustering Analysis...")
    cluster_metrics, clusters, rep_indices, outliers = perform_clustering(X_scaled, y, full_df)
    
    print("[CHECKPOINT] Saving Results...")
    full_df['cluster'] = clusters
    full_df['is_outlier'] = outliers == -1
    full_df['is_representative'] = False
    full_df.loc[rep_indices, 'is_representative'] = True
    
    print("[CHECKPOINT] Generating Final Results Comparison...")
    classification_results = pd.DataFrame({
        "KNN": knn_metrics,
        "Naïve Bayes": nb_metrics,
        "Random Forest": rf_metrics
    }).T
    
    logging.debug(f"Classification Performance:\n{classification_results}")
    
    classification_results.to_csv("classification_results.csv")
    
    logging.debug("Clustering Performance:")
    for metric, value in cluster_metrics.items():
        logging.debug(f"{metric}: {value:.3f}")
    
    with open("clustering_metrics.json", "w") as f:
        json.dump(cluster_metrics, f)
    
    # Ensure dates are in the correct format (YYYY-MM-DD) for CSV output
    full_df['snapshot_date'] = full_df['snapshot_date'].dt.strftime('%Y-%m-%d')
    full_df['album_release_date'] = full_df['album_release_date'].dt.strftime('%Y-%m-%d')
    
    # Replace NaN dates with empty string to preserve original data where invalid
    full_df['snapshot_date'] = full_df['snapshot_date'].fillna('')
    full_df['album_release_date'] = full_df['album_release_date'].fillna('')
    
    # Save to CSV with explicit encoding
    full_df.to_csv("dataset.csv", index=False, encoding='utf-8')
    
    print("[CHECKPOINT] All Results and Visualizations Saved Successfully!")

if __name__ == "__main__":
    main()