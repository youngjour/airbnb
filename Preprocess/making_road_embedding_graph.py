import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from cartoframes.viz import *
import os
import pickle
import utils


# 동 데이터 가져오기
def get_dong_data():
    dong_boundary = gpd.read_file('../emd/BND_ADM_DONG_PG.shp', encoding='CP949')
    dong = dong_boundary[dong_boundary['ADM_CD'].str.startswith('11')].copy()   # 서울시내 동 정보 추출
    dong['ADM_CD'] = dong['ADM_CD'].astype(str)
    
    dong = utils.set_dong424(dong)
    
    dong.to_crs(epsg=4326, inplace=True)
    dong = dong.drop(columns=['BASE_DATE', 'ADM_CD'])
    
    # 서울 경계 지도
    seoul_boundary = dong.unary_union
    
    return dong, seoul_boundary

# 구 데이터 가져오기
def get_gu_data():
    gu_boundary = gpd.read_file('../sig/sig.shp', encoding='CP949')
    gu = gu_boundary[gu_boundary['SIG_CD'].str.startswith('11')].copy()
    gu.set_crs(epsg=5179, inplace=True) # 미터 단위로 인식
    gu.to_crs(epsg=4326, inplace=True)  # 통상적인 위/경도로 변경
    
    return gu

def get_near_road_info():
    idx = pd.read_csv('../Data/Preprocessed_data/Dong/Human_flow.csv')
    near_road = pd.read_csv('../Data/Preprocessed_data/Dong/near_road_info_100m.csv')   # ====100m로 수정 필요!====
    
    idx = idx[['Reporting Month', 'Dong_name']]
    air_near_road = near_road.groupby(['Reporting Month','Dong_name']).agg({'num_nearby_nodes': 'sum'})
    air_near_road = air_near_road.reset_index(drop=False)
    
    air_near_road_mon = idx.merge(air_near_road, on=["Reporting Month", "Dong_name"], how="left").fillna(0)
    return air_near_road_mon

def get_seoul_road_data(seoul_boundary):
    # 서울시 전체 도로 네트워크 불러오기
    G_seoul = ox.graph_from_polygon(seoul_boundary, network_type='drive')
    
    nodes, edges = ox.graph_to_gdfs(G_seoul)
    new_edges = edges[['osmid', 'name', 'highway', 'oneway', 'length', 'geometry', 'bridge', 'tunnel']] # 사용할 정보만 가져오기
    
    # 1. highway 전처리 -> highway 열을 업데이트하여 단일 값으로 통일
    new_edges['highway'] = new_edges['highway'].apply(lambda x: utils.select_primary_highway(x) if isinstance(x, list) else x)
    
    # 2. bridge 전처리
    new_edges['bridge'] = new_edges['bridge'].apply(utils.categorize_bridge)
    
    # 3. tunnel 전처리
    new_edges['tunnel'] = new_edges['tunnel'].apply(utils.categorize_tunnel)
    
    return nodes, new_edges

# 인접 동 탐색
def find_neighbors(dong):
    neighbors = {}

    dong.to_crs(epsg=5179, inplace=True)

    # 각 동에 대해 다른 동들과 비교하여 인접 동 계산
    for idx, row in dong.iterrows():
        dong_name = row['ADM_NM']      # 현재 동의 이름
        current_geom = row['geometry'] # 현재 동의 지리정보
        friends = dong[(dong.geometry.intersects(current_geom)) & (dong['ADM_NM'] != dong_name)]['ADM_NM'].tolist()  # 인접 조건: 경계가 닿는 동들
        
        neighbors[dong_name] = friends
    
    dong.to_crs(epsg=4326, inplace=True)
    
    return neighbors

# 행정 단위별 도로 임베딩 생성
def get_region_embedding(region, new_edges, check):
    if check == 'dong':
        name = 'ADM_NM'
    else:
        name = 'SIG_KOR_NM'
    
    # 행정 단위별 도로 임베딩 저장
    region_embeddings = []
    for idx, row in region.iterrows():
        region_name = row[name]
        region_boundary = row['geometry']
    
        # 지역 경계에 포함된 도로 필터링
        roads_in_region = new_edges[new_edges['geometry'].within(region_boundary)]
        region_road_df = roads_in_region.reset_index(drop=True)
        
        # 지역별 도로 임베딩 생성
        region_embedding = utils.make_embedding(region_name, region_road_df)
        
        region_embeddings.append(region_embedding)
    
    region_embedding_df = pd.DataFrame(region_embeddings).fillna(0)
    
    # 각 행정 구역 면적 추가
    region_area = region.copy()
    region_area.to_crs(epsg=5179, inplace=True)
    region_area['area'] = region_area['geometry'].area
    region_area.rename(columns={name: 'name'}, inplace=True)
    
    region_embedding_df = region_embedding_df.merge(region_area[['name', 'area']], on='name', how='inner')
    
    return region_embedding_df

# 동 그래프 생성
def make_dong_graph(dong, neighbors, new_edges):
    # 동별 연결된 동과의 도로 개수를 그래프로 표현
    G = nx.Graph()

    dong.to_crs(epsg=5179, inplace=True)
    new_edges.to_crs(epsg=5179, inplace=True)

    for start, targets in neighbors.items():
        dong_boundary = dong.loc[dong['ADM_NM'] == start, 'geometry'].values[0]  # 시작동 경계

        # 시작 동을 노드로 추가
        G.add_node(start)

        # 이웃한 동들 도로 개수 계산
        for target in targets:
            target_boundary = dong.loc[dong['ADM_NM'] == target, 'geometry'].values[0]

            # 공유하는 도로의 개수 확인
            connecting_roads = new_edges[new_edges['geometry'].intersects(dong_boundary) &
                                         new_edges['geometry'].intersects(target_boundary)]
            connected_counts = len(connecting_roads)  # 공유 도로 개수

            if connected_counts == 0:
                connected_counts = 0.1  # 인접성 정보 유지

            # 대상 동을 노드로 추가
            G.add_node(target)

            # 그래프에 엣지 추가 (가중치: 도로 개수)
            G.add_edge(start, target, weight=connected_counts)

    # 그래프 저장
    output_file = os.path.join('../Data/Graph', "road_graph.gpickle")
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)

# dong_embeddings, gu_embeddings 정규화 함수 -> 일단 적용하지 말자..
def normalize_embeddings(df, id_column_name):
    # 첫 번째 열을 제외한 나머지 열 정규화
    columns_to_normalize = df.columns.difference([id_column_name])  # ID 열 제외
    df_normalized = df.copy()  # 원본 데이터 보존
    
    for col in columns_to_normalize:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val == 0:  # 값이 모두 동일하면 0으로 설정
            df_normalized[col] = 0
        else:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def main():
    # 동, 구 데이터 불러오기기
    dong, seoul_boundary = get_dong_data()
    gu = get_gu_data()
    near_road_info = get_near_road_info()
    
    print('데이터 불러오기 성공!')
    
    # 서울시내 도로 정보 가져오기
    nodes, new_edges = get_seoul_road_data(seoul_boundary)
    print('도로 데이터 불러오기 성공!')
    
    # 동, 구별 임베딩 생성
    dong_embeddings = get_region_embedding(dong, new_edges, 'dong')
    gu_embeddings = get_region_embedding(gu, new_edges, 'gu')
    
    dong_embeddings.rename(columns={'name': 'Dong_name'}, inplace=True)
    gu_embeddings.rename(columns={'name': 'Gu_name'}, inplace=True)
    
    # min-max 정규화 실행
    #dong_embeddings = normalize_embeddings(dong_embeddings, 'Dong_name')
    #gu_embeddings = normalize_embeddings(gu_embeddings, 'Gu_name')
    
    mon_dong_embeddings = near_road_info.merge(dong_embeddings, on=['Dong_name'])
    
    print(dong_embeddings.head())
    print(gu_embeddings.head())
    
    # 임베딩 저장
    mon_dong_embeddings.to_csv('../Data/Preprocessed_data/Dong/Road_Embeddings_plus.csv', index=False)
    gu_embeddings.to_csv('../Data/Preprocessed_data/Gu/Road_Embeddings.csv', index=False)
    print('임베딩 저장 완료!')
    
    # 동별 그래프 생성
    neighbors = find_neighbors(dong)
    make_dong_graph(dong, neighbors, new_edges)
    print('동 그래프 생성 완료!')
    
if __name__ == "__main__":
    main()
