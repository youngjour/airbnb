import pandas as pd
import numpy as np

def read_human_flow():
    df = pd.read_csv('../Data/Raw_data/Human_flow.csv')
    
    # 열 이름 영문 변경
    translated_columns = [
        'Total Population',
        'Male Population (0–9 years)', 'Male Population (10–14 years)', 'Male Population (15–19 years)',
        'Male Population (20–24 years)', 'Male Population (25–29 years)', 'Male Population (30–34 years)',
        'Male Population (35–39 years)', 'Male Population (40–44 years)', 'Male Population (45–49 years)',
        'Male Population (50–54 years)', 'Male Population (55–59 years)', 'Male Population (60–64 years)',
        'Male Population (65–69 years)', 'Male Population (70+ years)', 'Female Population (0–9 years)',
        'Female Population (10–14 years)', 'Female Population (15–19 years)', 'Female Population (20–24 years)',
        'Female Population (25–29 years)', 'Female Population (30–34 years)', 'Female Population (35–39 years)',
        'Female Population (40–44 years)', 'Female Population (45–49 years)', 'Female Population (50–54 years)',
        'Female Population (55–59 years)', 'Female Population (60–64 years)', 'Female Population (65–69 years)',
        'Female Population (70+ years)', 'Total Long-Term Foreigner Population',
        'Long-Term Foreigner Population (Chinese)', 'Long-Term Foreigner Population (Non-Chinese)',
        'Total Short-Term Foreigner Population', 'Short-Term Foreigner Population (Chinese)',
        'Short-Term Foreigner Population (Non-Chinese)'
    ]
    
    cols = list(df.columns[:2]) + translated_columns + ['Gu_name']
    df.columns = cols
    
    return df

def normalize_to_standard_normal(df):
    """
    데이터프레임의 각 열을 정규분포로 정규화 (z-score normalization)
    """
    df_normalized = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # 숫자형 데이터에 대해서만 적용
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:  # 표준편차가 0인 경우 처리
                df_normalized[col] = 0
            else:
                df_normalized[col] = (df[col] - mean) / std
    return df_normalized

def get_region_df(df, region_name):
    df = df.copy()
    df = df.drop(columns=[region_name])
    return df

def main():
    df = read_human_flow()
    
    dong_df = df.copy()
    gu_df = df.copy()
    
    dong_df = get_region_df(df, 'Gu_name')
    gu_df = get_region_df(df, 'Dong_name')
    
    # 구단위 생활인구 종합
    gu_df = gu_df.groupby(['Reporting Month', 'Gu_name'], as_index=False).sum()
    
    # 일단 정규화는 생략략
    #dong_df = normalize_to_standard_normal(dong_df)
    #gu_df = normalize_to_standard_normal(gu_df)
    
    # 데이터 저장
    dong_df.to_csv('../Data/Preprocessed_data/Dong/Human_flow.csv', index=False)
    gu_df.to_csv('../Data/Preprocessed_data/Gu/Human_flow.csv', index=False)
    
if __name__ == "__main__":
    main()
    