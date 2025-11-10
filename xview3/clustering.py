import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

# Lista dei campi da mantenere e come trattarli nel merge
COLUMNS_TO_KEEP = [
    'vessel_length_m', 'detect_scene_row', 'detect_scene_column',
    'scene_id', 'objectness_p', 'is_vessel_p', 'is_fishing_p',
    'objectness_threshold', 'longitude', 'latitude', 'is_vessel',
    'is_fishing'
]

# Funzione per calcolare la matrice di distanza 
def compute_distance_matrix(df):
    """Calcola la matrice delle distanze in metri tra tutti i punti."""
    coords = df[['latitude', 'longitude']].values
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = great_circle(coords[i], coords[j]).meters
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            
    return distance_matrix

# --- 2. Funzione di Merging In-Place ---

def merge_ships_in_place(df: pd.DataFrame, max_cluster_dist_m: float = 150, 
                         length_plausibility_factor: float = 3.0) -> pd.DataFrame:
    """
    Raggruppa le rilevazioni frammentate, filtra per plausibilità e modifica
    il DataFrame, rimuovendo i frammenti e aggiungendo la nave mergiata.

    Restituisce un nuovo DataFrame pulito.
    """
    
    # Lavora su una copia per non alterare l'originale finché non è completo
    df_copy = df.copy().reset_index(drop=True)
    
    if df_copy.empty:
        return pd.DataFrame(columns=COLUMNS_TO_KEEP)

    # 1. DBSCAN Iniziale (Clustering "Ampio")
    print(f"-> Esecuzione DBSCAN con max_cluster_dist_m={max_cluster_dist_m}m...")
    distance_matrix = compute_distance_matrix(df_copy)
    
    # DBSCAN non dovrebbe considerare righe isolate come "rumore" (-1)
    # se vogliamo considerare tutte le righe come candidati validi al merge o singole navi
    db = DBSCAN(eps=max_cluster_dist_m, min_samples=1, metric="precomputed").fit(distance_matrix)
    df_copy['cluster_id'] = db.labels_

    
    # Lista degli indici da eliminare dopo l'iterazione
    indices_to_drop = []
    
    # Lista delle nuove righe da aggiungere
    new_rows = []
    
    for cluster_id, group in df_copy.groupby('cluster_id'):
        
        # Ignora i cluster di "rumore" (-1) o i cluster di singole rilevazioni (vengono gestiti sotto)
        if len(group) == 1 or cluster_id == -1:
             continue 

        # --- 2. Calcolo della Plausibilità ---
        
        # A. Lunghezza Predetta Media (L_mean). Usiamo il campo vessel_length_m
        L_mean = group['vessel_length_m'].mean()
        
        # B. Lunghezza Geometrica (L_geom): Distanza massima tra tutti i centri nel cluster
        cluster_coords = group[['latitude', 'longitude']].values
        max_dist = 0
        for i in range(len(cluster_coords)):
            for j in range(i + 1, len(cluster_coords)):
                dist = great_circle(cluster_coords[i], cluster_coords[j]).meters
                if dist > max_dist:
                    max_dist = dist
        L_geom = max_dist

        # C. Criterio di Filtro
        ratio = L_geom / L_mean if L_mean > 0 else float('inf')
        
        if ratio <= length_plausibility_factor:
            # ACCETTA: Il cluster è plausibile, esegui il merge
            print(f"  Cluster {cluster_id}: Accettato (Ratio L_geom/L_mean = {ratio:.2f}). Merged.")
            
            # 1. Raccogli gli indici da eliminare
            indices_to_drop.extend(group.index.tolist())
            
            # 2. Crea la nuova riga mergiata
            new_row = {}
            
            # Campi da aggregare/mediare/scegliere
            new_row['longitude'] = group['longitude'].mean()
            new_row['latitude'] = group['latitude'].mean()
            
            # La lunghezza finale è L_geom
            new_row['vessel_length_m'] = L_geom
            
            # Per i campi non aggregabili, usiamo la moda (il valore più frequente) o il primo valore
            new_row['scene_id'] = group['scene_id'].mode()[0] if not group['scene_id'].empty else None
            
            # Usa il massimo per i punteggi di probabilità/objectness
            new_row['objectness_p'] = group['objectness_p'].max()
            new_row['is_vessel_p'] = group['is_vessel_p'].max()
            new_row['is_fishing_p'] = group['is_fishing_p'].max()
            
            # Per i campi booleani/categorici, usiamo la moda
            new_row['is_vessel'] = group['is_vessel'].mode()[0] if not group['is_vessel'].empty else None
            new_row['is_fishing'] = group['is_fishing'].mode()[0] if not group['is_fishing'].empty else None

            # Per le coordinate scena (non utili dopo il merge, usiamo la media o un placeholder)
            new_row['detect_scene_row'] = group['detect_scene_row'].mean()
            new_row['detect_scene_column'] = group['detect_scene_column'].mean()
            
            # Threshold: usa il valore più stringente (minimo) o la moda
            new_row['objectness_threshold'] = group['objectness_threshold'].min() 

            new_rows.append(new_row)
        else:
            # SCARTA: Cluster implausibile. Le righe rimarranno separate nel DataFrame finale.
            print(f"  Cluster {cluster_id}: Scartato (Ratio L_geom/L_mean = {ratio:.2f} > {length_plausibility_factor}). Divisione.")
    
    # --- 3. Modifica del DataFrame ---
    
    # 1. Elimina i frammenti originali
    df_merged = df_copy.drop(indices_to_drop)

    # 2. Aggiungi le nuove righe mergiate
    df_new_ships = pd.DataFrame(new_rows)
    df_merged = pd.concat([df_merged, df_new_ships], ignore_index=True)

    # Pulisci e riordina le colonne
    return df_merged[COLUMNS_TO_KEEP]

# --- 3. Esempio di Utilizzo ---
if __name__ == "__main__":

    df_detections = pd.read_csv("Predictions.csv")

    # Parametri scelti per dimostrare la divisione:
    # max_cluster_dist_m = 150m (abbastanza per unire tutti e 4 i punti nell'esempio)
    # length_plausibility_factor = 1.0 (vincolo stretto: la distanza massima non può superare la lunghezza media predetta)
    df_merged = merge_ships_in_place(df_detections, 
                                            max_cluster_dist_m=400, 
                                            length_plausibility_factor=1.5) 

    print("\n--- Rilevazioni (Merged) ---")
    print(df_merged)