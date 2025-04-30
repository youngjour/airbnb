

def set_dong424(dong):
    # 중복이름 동 방지
    dong.loc[(dong['ADM_NM']=='신사동') & (dong['ADM_CD']=='11230510'), 'ADM_NM'] = '신사동(강남)'
    dong.loc[(dong['ADM_NM']=='신사동') & (dong['ADM_CD']=='11210680'), 'ADM_NM'] = '신사동(관악)'
    
    # 424개의 동으로 맞추기
    # ==================================================================================================
    # 상일동 처리
    sangil_union = dong[dong["ADM_NM"].isin(["상일1동", "상일2동"])]["geometry"].unary_union
    # 상일1동 -> 상일동
    dong.loc[dong['ADM_NM']=='상일1동', 'ADM_NM'] = '상일동'
    dong.loc[dong['ADM_NM']=='상일동', 'geometry'] = sangil_union
    # 상일2동 제거
    dong = dong[~dong["ADM_NM"].isin(["상일2동"])]

    # 개포3동 -> 일원2동 변경
    dong.loc[dong['ADM_NM']=='개포3동', 'ADM_NM'] = '일원2동'

    # 항동을 오류2동에 포함
    oru2_union = dong[dong["ADM_NM"].isin(["항동", "오류2동"])]["geometry"].unary_union
    # 오류2동 좌표 변경
    dong.loc[dong['ADM_NM']=='오류2동', 'geometry'] = oru2_union
    # 항동 제거
    dong = dong[~dong["ADM_NM"].isin(["항동"])]
    
    return dong


# 여러 개의 highway 값을 가진 행에 대해 우선순위가 높은 값 선택
def select_primary_highway(highway):
    highway_priority = {
        'motorway': 1,
        'motorway_link': 2,
        'trunk': 3,
        'trunk_link': 4,
        'primary': 5,
        'primary_link': 6,
        'secondary': 7,
        'secondary_link': 8,
        'tertiary': 9,
        'tertiary_link': 10,
        'unclassified': 11,
        'residential': 12,
        'living_street': 13,
        'service': 14,
        'pedestrian': 15,
        'track': 16,
        'path': 17
    }
    # highway가 리스트 형식일 경우
    if isinstance(highway, list):
        # 우선순위에 따라 정렬하고 가장 우선순위가 높은 값 반환
        return sorted(highway, key=lambda x: highway_priority.get(x, float('inf')))[0]
    return highway  # 단일 값이면 그대로 반환

def categorize_bridge(value):
    if isinstance(value, list):
        # 리스트에 'viaduct'가 있으면 2 (고가도로), 'yes'만 있으면 1 (일반 다리)
        if 'viaduct' in value:
            return 2
        elif 'yes' in value:
            return 1
    elif value == 'viaduct':
        return 2  # 고가도로
    elif value == 'yes':
        return 1  # 일반 다리
    return 0  # 다리 정보 없음

def categorize_tunnel(value):
    if value == 'building_passage':
        return 2  # 건물 통과 구간
    elif value == 'yes':
        return 1  # 일반 터널
    return 0  # 터널 정보 없음

def make_embedding(region_name, region_road_df):
    # 1. 도로 개수
    road_count = len(region_road_df)
    
    # 2. highway 값별 개수
    #print(dong_name)
    #print(roads_df[roads_df['highway'].apply(lambda x: isinstance(x, list))])
    highway_counts = region_road_df['highway'].value_counts().to_dict()
    
    # 3. 총 길이
    total_length = region_road_df['length'].sum()
    
    # 4. 터널 개수 (일반 터널과 건물 통과 구간)
    tunnel_counts = region_road_df['tunnel'].value_counts().reindex([1, 2], fill_value=0).to_dict()
    
    # 5. 다리 개수 (일반 다리와 고가도로)
    bridge_counts = region_road_df['bridge'].value_counts().reindex([1, 2], fill_value=0).to_dict()
    
    # 6. 임베딩 벡터 구성
    embedding = {
        'name': region_name,
        'road_count': road_count,
        'total_length': total_length,
        'tunnel_count_general': tunnel_counts.get(1, 0),
        'tunnel_count_building_passage': tunnel_counts.get(2, 0),
        'bridge_count_general': bridge_counts.get(1, 0),
        'bridge_count_viaduct': bridge_counts.get(2, 0)
    }
    
    # highway별 개수를 임베딩에 추가
    for highway_type, count in highway_counts.items():
        embedding[f'highway_count_{highway_type}'] = count
        
    return embedding