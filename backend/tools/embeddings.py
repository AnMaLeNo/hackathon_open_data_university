import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil utilisé : {device.upper()}")

print("Chargement du modèle dans la VRAM...")
model = SentenceTransformer("BAAI/bge-m3", device=device)
# On ne met PLUS de limite de tokens. Le modèle lira vos textes en entier.

# 2. Chargement des données
print("Chargement du fichier parquet...")
df = pd.read_parquet("./backend/database/reactions.parquet")

def clean_text(x):
    if isinstance(x, str):
        return x
    return ""

textes = [clean_text(x) for x in df["comment"].tolist()]
ids = df['id'].astype(str).tolist()

# 3. Paramètres du "Pas à Pas"
TAILLE_BLOC_SAUVEGARDE = 500  # On sauvegarde un fichier toutes les 100 lignes
BATCH_SIZE_GPU = 2            # On envoie les textes 2 par 2 dans la RTX 3080 pour ne pas la saturer

print("\n🚀 Démarrage de l'encodage étape par étape...")

# 4. La boucle de traitement sécurisée
for i in range(0, len(textes), TAILLE_BLOC_SAUVEGARDE):
    
    # On isole un bloc de 100 textes
    fin = min(i + TAILLE_BLOC_SAUVEGARDE, len(textes))
    bloc_textes = textes[i:fin]
    bloc_ids = ids[i:fin]
    
    nom_fichier_sortie = f"embeddings_sauvegarde_{i}_a_{fin}.parquet"
    
    # Si le fichier existe déjà (ex: suite à un plantage précédent), on le saute !
    if os.path.exists(nom_fichier_sortie):
        print(f"⏩ Le bloc {i} à {fin} est déjà fait, on passe...")
        continue
        
    print(f"\nTraitement des lignes {i} à {fin}...")
    
    # Encodage avec un tout petit batch_size pour protéger la mémoire
    bloc_embeddings = model.encode(bloc_textes, batch_size=BATCH_SIZE_GPU, show_progress_bar=True)
    
    # Sauvegarde immédiate sur le disque
    df_temp = pd.DataFrame({
        'id': bloc_ids,
        'embedding': bloc_embeddings.tolist()
    })
    
    df_temp.to_parquet(nom_fichier_sortie)
    print(f"✅ Sauvegardé : {nom_fichier_sortie}")

print("\n🎉 Tout est terminé avec succès !")