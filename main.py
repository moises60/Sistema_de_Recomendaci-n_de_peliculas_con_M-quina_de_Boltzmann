import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

from models.rbm import RBM
from utils.data_preprocessing import load_ratings, load_user_features, convert_optimized, binarize
from utils.evaluation import compute_loss
from Recommender import get_user_personal_data, get_user_ratings, recommend_movies

import matplotlib.pyplot as plt

# ===============================
# 1. Configuración del Dispositivo
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ===============================
# 2. Cargar y Preprocesar los Datos
# ===============================

# 2.1. Cargar las valoraciones
ratings = load_ratings('data/ml-1m/ratings.dat')

# 2.2. Obtener el número de usuarios y películas
nb_users = ratings['UserID'].nunique()
nb_movies = ratings['MovieID'].nunique()

print(f"Número de usuarios: {nb_users}")
print(f"Número de películas: {nb_movies}")

# 2.3. Mapear MovieIDs a índices consecutivos
unique_movie_ids = ratings['MovieID'].unique()
movie_id_to_index = {movie_id: index for index, movie_id in enumerate(unique_movie_ids)}

# 2.4. Cargar y procesar datos de usuarios
user_features_tensor, num_user_features = load_user_features('data/ml-1m/users.dat')
user_features_tensor = user_features_tensor.to(device)

# 2.5. Crear división de entrenamiento y prueba
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# 2.6. Convertir los datos
training_set = convert_optimized(train_data, nb_users, nb_movies, movie_id_to_index)
test_set = convert_optimized(test_data, nb_users, nb_movies, movie_id_to_index)

# 2.7. Binarizar las valoraciones
training_set = binarize(training_set).to(device)
test_set = binarize(test_set).to(device)

# 2.8. Concatenar características de usuario a las valoraciones
training_set = torch.cat((training_set, user_features_tensor), dim=1)
test_set = torch.cat((test_set, user_features_tensor), dim=1)

# ===============================
# 3. Definir la RBM y el Optimizador
# ===============================

nv = nb_movies + num_user_features  # Número de unidades visibles
nh = 200                            # Número de unidades ocultas
batch_size = 128
lr = 0.005

rbm = RBM(nv, nh, device, lr).to(device)

# Regularización L2
weight_decay = 0.0001

optimizer = optim.SGD(rbm.parameters(), lr=rbm.lr, weight_decay=weight_decay)

# ===============================
# 4. Preparar los DataLoaders
# ===============================

train_dataset = TensorDataset(training_set)
test_dataset = TensorDataset(test_set)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===============================
# 5. Entrenamiento de la RBM
# ===============================

nb_epoch = 50
loss_per_epoch = []
test_loss_per_epoch = []

num_visible_ratings = nb_movies  # Número de columnas de valoraciones

for epoch in range(1, nb_epoch + 1):
    rbm.train()
    training_loss = 0
    s = 0.

    for batch in train_loader:
        v0 = batch[0].to(device)
        vk = v0.clone()

        # Muestrear las unidades ocultas
        ph0, _ = rbm.sample_h(v0)

        # Contrastive Divergence
        for k in range(1):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            # Mantener características de usuario sin cambios
            vk[:, num_visible_ratings:] = v0[:, num_visible_ratings:]
            # Mantener valoraciones no observadas sin cambios
            vk[:, :num_visible_ratings] = torch.where(
                v0[:, :num_visible_ratings] < 0,
                v0[:, :num_visible_ratings],
                vk[:, :num_visible_ratings]
            )

        phk, _ = rbm.sample_h(vk)

        # Zero grad
        optimizer.zero_grad()

        # Calcular y asignar gradientes
        rbm.train_rbm(v0, vk, ph0, phk)

        # Actualizar parámetros
        optimizer.step()

        # Calcular pérdida
        loss_mae = compute_loss(v0, vk, num_visible_ratings)
        training_loss += loss_mae
        s += 1.

    average_loss = training_loss / s if s > 0 else 0
    loss_per_epoch.append(average_loss)

    # Evaluación en el conjunto de prueba
    rbm.eval()
    testing_loss = 0
    s_test = 0.

    with torch.no_grad():
        for batch in test_loader:
            v = batch[0].to(device)
            vt = v.clone()
            _, h = rbm.sample_h(v)
            _, v_rec = rbm.sample_v(h)
            # Mantener características de usuario sin cambios
            v_rec[:, num_visible_ratings:] = v[:, num_visible_ratings:]
            # Mantener valoraciones no observadas sin cambios
            v_rec[:, :num_visible_ratings] = torch.where(
                vt[:, :num_visible_ratings] < 0,
                vt[:, :num_visible_ratings],
                v_rec[:, :num_visible_ratings]
            )
            loss_mae_test = compute_loss(vt, v_rec, num_visible_ratings)
            testing_loss += loss_mae_test
            s_test += 1.

    average_test_loss = testing_loss / s_test if s_test > 0 else 0
    test_loss_per_epoch.append(average_test_loss)

    print(f"Epoch: {epoch}, Loss MAE: {average_loss:.4f}, Test Loss MAE: {average_test_loss:.4f}")

# ===============================
# 6. Visualización de la Pérdida
# ===============================

plt.figure(figsize=(10,5))
plt.plot(range(1, nb_epoch + 1), loss_per_epoch, marker='o', label='Pérdida de Entrenamiento MAE')
plt.plot(range(1, nb_epoch + 1), test_loss_per_epoch, marker='x', label='Pérdida de Prueba MAE')
plt.xlabel('Época')
plt.ylabel('Pérdida MAE')
plt.title('Pérdida de Entrenamiento y Prueba por Época')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 7. Guardar el Modelo Entrenado
# ===============================

model_path = 'models/trained_rbm.pth'
torch.save(rbm.state_dict(), model_path)
print(f'Modelo guardado en {model_path}')

# ===============================
# 8. Generar Recomendaciones para un Usuario
# ===============================

from Recommender import get_user_personal_data, get_user_ratings, recommend_movies

# Cargar títulos de películas
try:
    movies = pd.read_csv(
        'data/ml-1m/movies.dat',
        sep='::',
        header=None,
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'  # En el de 32 M es distinto 
    )
    print("Archivo 'movies.dat' cargado exitosamente.")
except FileNotFoundError:
    print("El archivo 'movies.dat' no se encontró en la ruta especificada.")
    exit(1)
except pd.errors.ParserError as pe:
    print("Error de análisis al leer el archivo 'movies.dat':")
    print(pe)
    exit(1)
except Exception as e:
    print("Ocurrió un error al leer el archivo 'movies.dat':")
    print(e)
    exit(1)

#Filtrar películas para incluir solo aquellas en movie_id_to_index
movies = movies[movies['MovieID'].isin(movie_id_to_index.keys())]

#Obtener datos personales del usuario
user_features_tensor = get_user_personal_data().to(device)

#Obtener valoraciones del usuario
user_ratings_tensor = get_user_ratings(movies[['MovieID', 'Title']], movie_id_to_index).to(device)

#Binarizar las valoraciones del usuario
user_ratings_tensor = binarize(user_ratings_tensor)

#Concatenar valoraciones y características
user_input = torch.cat((user_ratings_tensor, user_features_tensor), dim=1)

#Recomendar películas
num_recommendations = 10
recommended_movies_indices = recommend_movies(
    rbm, user_input, num_recommendations, num_visible_ratings, device
)

#Mapear índices a MovieIDs
index_to_movie_id = {index: movie_id for movie_id, index in movie_id_to_index.items()}

#Obtener MovieIDs recomendadas
recommended_movie_ids = [index_to_movie_id[idx] for idx in recommended_movies_indices]

#Obtener títulos de las películas recomendadas
recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]

print("\nLas siguientes películas son recomendadas para ti:")
for idx, row in recommended_movies.iterrows():
    print(f"- {row['Title']} ({row['Genres']})")
