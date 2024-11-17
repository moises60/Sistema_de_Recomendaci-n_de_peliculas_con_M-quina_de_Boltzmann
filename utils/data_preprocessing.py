import pandas as pd
import torch
import scipy.sparse as sp

def load_ratings(path):
    """
    Carga las valoraciones desde un archivo.

    Args:
        path (str): Ruta al archivo de valoraciones.

    Returns:
        ratings (pd.DataFrame): DataFrame con las valoraciones.
    """
    ratings = pd.read_csv(
        path,
        sep='::',
        header=None,
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    return ratings

def load_user_features(path):
    """
    Carga y procesa las características de los usuarios.

    Args:
        path (str): Ruta al archivo de usuarios.

    Returns:
        user_features_tensor (torch.Tensor): Tensor con las características de los usuarios.
        num_user_features (int): Número de características por usuario.
    """
    users = pd.read_csv(
        path,
        sep='::',
        header=None,
        engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    )

    # Codificar género
    gender_dummies = pd.get_dummies(users['Gender'], prefix='Gender')

    # Codificar edad
    age_dummies = pd.get_dummies(users['Age'], prefix='Age')

    # Codificar ocupación
    occupation_dummies = pd.get_dummies(users['Occupation'], prefix='Occ')

    # Combinar características de usuario
    user_features = pd.concat([gender_dummies, age_dummies, occupation_dummies], axis=1)

    # Asegurar que UserID comienza desde 0
    users['UserID'] = users['UserID'] - 1
    user_features = user_features.values  # Convertir a numpy array

    # Convertir a tensor
    user_features_tensor = torch.FloatTensor(user_features)
    num_user_features = user_features.shape[1]

    return user_features_tensor, num_user_features

def convert_optimized(data, nb_users, nb_movies, movie_id_to_index):
    """
    Convierte los datos en una matriz donde las filas representan usuarios
    y las columnas representan películas. Las valoraciones no observadas se asignan a 0.

    Args:
        data (pd.DataFrame): DataFrame con las valoraciones.
        nb_users (int): Número total de usuarios.
        nb_movies (int): Número total de películas.
        movie_id_to_index (dict): Diccionario para mapear MovieID a índices.

    Returns:
        dense_matrix (torch.Tensor): Tensor denso con las valoraciones.
    """
    data = data.copy()
    data['UserID'] = data['UserID'] - 1  # Ajustar índices de usuarios
    data['MovieIndex'] = data['MovieID'].map(movie_id_to_index)

    # Filtrar entradas válidas
    data = data.dropna(subset=['MovieIndex'])
    data['MovieIndex'] = data['MovieIndex'].astype(int)

    # Construir la matriz dispersa
    rows = data['UserID'].astype(int)
    cols = data['MovieIndex']
    ratings = data['Rating']
    sparse_matrix = sp.coo_matrix((ratings, (rows, cols)), shape=(nb_users, nb_movies))

    # Convertir a tensor denso
    dense_matrix = torch.FloatTensor(sparse_matrix.toarray())

    return dense_matrix

def binarize(tensor):
    """
    Binariza las valoraciones: 1 para valoraciones >= 3, 0 para valoraciones 1-2, -1 para no observadas.

    Args:
        tensor (torch.Tensor): Tensor con las valoraciones originales.

    Returns:
        tensor (torch.Tensor): Tensor binarizado.
    """
    tensor = tensor.clone()
    tensor[tensor == 0] = -1
    tensor[tensor == 1] = 0
    tensor[tensor == 2] = 0
    tensor[tensor >= 3] = 1
    return tensor

def load_movies(path):
    """
    Carga los datos de las películas.

    Args:
        path (str): Ruta al archivo de películas.

    Returns:
        movies (pd.DataFrame): DataFrame con los datos de las películas.
    """
    movies = pd.read_csv(
        path,
        sep='::',
        header=None,
        engine='python',
        names=['MovieID', 'Title', 'Genres']
    )
    return movies