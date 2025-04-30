import pandas as pd

def get_data():
    df = pd.read_csv('../Data/Raw_data/AirBnB_data.csv')
    return df

def make_embedding(df):
    raw_cols = ['Reporting Month', 'Dong_name', 'Gu_name', 'Reservation Days', 'Available Days', 
            'Blocked Days', 'Property Type', 'Listing Type', 'Currency Native', 'Number of Reviews', 
            'Bedrooms', 'Bathrooms', 'Max Guests', 'Response Rate', 'Airbnb Response Time (Text)', 'Airbnb Superhost', 
            'Cancellation Policy', 'Security Deposit (USD)', 'Cleaning Fee (USD)', 'Extra People Fee (USD)', 'Published Nightly Rate (USD)', 
            'Published Monthly Rate (USD)', 'Published Weekly Rate (USD)', 'Check-in Time', 'Checkout Time', 'Minimum Stay', 
            'Number of Photos', 'Instantbook Enabled', 'Latitude', 'Longitude',  'Pets Allowed', 'Integrated Property Manager']
    
    raw_df = df[raw_cols]
    
    # 제외할 열을 제외한 나머지 열을 one-hot 인코딩
    numeric_columns = raw_df.select_dtypes(include=['number']).columns
    excluded_columns = ['Reporting Month', 'Dong_name', 'Gu_name']
    excluded_columns.extend(numeric_columns)

    encoded_df = pd.get_dummies(raw_df.drop(columns=excluded_columns), drop_first=True)

    # 제외한 열을 다시 추가
    raw_embeddings = pd.concat([raw_df[excluded_columns], encoded_df], axis=1)
    
    # 데이터 분리
    dong_df = raw_embeddings.drop(columns=['Gu_name'])
    gu_df = raw_embeddings.drop(columns=['Dong_name'])

    dong_names = list(dong_df['Dong_name'].unique())
    dong_names.append('상계8동')
    gu_names = list(gu_df['Gu_name'].unique())

    dong_embedding = dong_df.groupby(['Reporting Month', 'Dong_name']).mean().reset_index()
    gu_embedding = gu_df.groupby(['Reporting Month', 'Gu_name']).mean().reset_index()

    date_range = pd.date_range(start='2017-01-01', end='2022-07-01', freq='MS')
    date_range = date_range.strftime('%Y-%m-%d')

    full_index = pd.MultiIndex.from_product([date_range, dong_names], names=['Reporting Month', 'Dong_name'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    merged_df = pd.merge(full_df, dong_embedding, on=['Reporting Month', 'Dong_name'], how='left')
    merged_df.fillna(0, inplace=True)

    # 저장
    merged_df.to_csv('../Data/Preprocessed_data/Dong/AirBnB_raw.csv', index=False)
    gu_embedding.to_csv('../Data/Preprocessed_data/Gu/AirBnB_raw.csv', index=False)
    
def main():
    df = get_data()
    make_embedding(df)
    
if __name__ == "__main__":
    main()