import pandas as pd
from haversine import haversine
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np

# Função para calcular a distância usando Haversine
def calcular_distancia(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2))

def remover_quebras_de_linha(texto):
    if isinstance(texto, str):
        return texto.replace('\r\n', ' ').strip()
    return texto

def calcular_distancia_pudo_vendedor(df_vendedores, df_pudos):
    # Remover linhas onde kyc_id é vazio
    df_vendedores = df_vendedores.dropna(subset=['kyc_id'])

    # Verificar e converter as colunas de latitude e longitude para tipo numérico
    df_vendedores[["latitude", "longitude"]] = df_vendedores[
        ["latitude", "longitude"]
    ].apply(pd.to_numeric, errors="coerce")
    df_pudos[["latitude", "longitude"]] = df_pudos[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remover linhas com valores não finitos
    df_vendedores = df_vendedores[np.isfinite(df_vendedores['latitude']) & np.isfinite(df_vendedores['longitude'])]
    df_pudos = df_pudos[np.isfinite(df_pudos['latitude']) & np.isfinite(df_pudos['longitude'])]

    # Criar KDTree para os pontos de coleta
    pontos_coleta_tree = cKDTree(df_pudos[["latitude", "longitude"]].values)

    # Lista para armazenar os resultados
    dados_finais = []

    # Iterar sobre cada linha do dataframe de vendedores
    for index_vendedor, row_vendedor in tqdm(
        df_vendedores.iterrows(), total=len(df_vendedores), desc="Calculando distâncias"
    ):
        lat_vendedor, lon_vendedor = (
            row_vendedor["latitude"],
            row_vendedor["longitude"],
        )

        # Consultar KDTree para encontrar todos os pontos de coleta mais próximos do vendedor
        _, indice_ponto_mais_proximo = pontos_coleta_tree.query(
            [lat_vendedor, lon_vendedor], k=1
        )

        # Encontrar o ponto mais próximo entre os pontos encontrados
        ponto_proximo = df_pudos.iloc[indice_ponto_mais_proximo]

        # Calcular a distância entre o vendedor e o ponto mais próximo
        distancia_proximo = calcular_distancia(
            lat_vendedor,
            lon_vendedor,
            ponto_proximo["latitude"],
            ponto_proximo["longitude"],
        )

        # Adicionar informações à lista, removendo quebras de linha de campos de texto
        if distancia_proximo <= 900.5:
            dados_finais.append(
                {
                    "distancia_mais_proxima": distancia_proximo,
                    "kyc_id": row_vendedor["kyc_id"],
                    "kyc_status": row_vendedor["kyc_status"],
                    "Address": row_vendedor["Address"],
                    "cidade": remover_quebras_de_linha(row_vendedor["City"]),
                    "estado": remover_quebras_de_linha(row_vendedor["State"]),
                    "cnpj": row_vendedor["cnpj"],
                    "latitude_point": row_vendedor["latitude"],
                    "longitude_point": row_vendedor["longitude"],
                    "ado_4w": ponto_proximo["ado_4w"],
                    "latitude_green": ponto_proximo["latitude"],
                    "longitude_green": ponto_proximo["longitude"],
                    "Region": ponto_proximo["Region"],
                    "region_id": ponto_proximo["region_id"],
                    "gf_lower": ponto_proximo["gf_lower"]
                }
        )

    # Criar DataFrame final a partir da lista de dados
    df_final = pd.DataFrame(dados_finais)

    return df_final
