"""
Generate Road Network LLM prompts from Road_Embeddings_with_flow.csv

Based on Hongju's make_additional_llm_prompt.ipynb
"""

import pandas as pd
import os

print("="*80)
print("ROAD NETWORK PROMPT GENERATION")
print("="*80)

# Load road data
file_path = '../../../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv'
print(f"\nLoading road network data from: {file_path}")
road = pd.read_csv(file_path)
print(f"[OK] Loaded {len(road)} rows x {len(road.columns)} columns")

# Create prompt dataframe
road_prompt_df = road[['Reporting Month', 'Dong_name']].copy()
road_prompt = []

print("\nGenerating road network prompts...")

for idx, row in road.iterrows():
    month = row['Reporting Month']
    dong = row['Dong_name']

    prompt = (
        f"[{month} | {dong}] Road and Transportation Overview:\n"
        f"- Number of road nodes near AirBnBs: {row.iloc[2]}\n"
        f"- Total number of roads in the dong: {row.iloc[3]}, Total length: {row.iloc[4]}\n"
        f"- Number of tunnels in the dong: {row.iloc[5]} general, {row.iloc[6]} building passage\n"
        f"- Number of bridges in the dong: {row.iloc[7]} general, {row.iloc[8]} viaducts\n"
        f"- Road types:\n"
        f"  • Residential: {row.iloc[9]}\n"
        f"  • Primary: {row.iloc[10]}\n"
        f"  • Tertiary: {row.iloc[11]}\n"
        f"  • Living streets: {row.iloc[12]}\n"
        f"  • Busways: {row.iloc[13]}\n"
        f"  • Secondary: {row.iloc[16]}\n"
        f"  • Trunk: {row.iloc[19]}\n"
        f"  • Motorway: {row.iloc[24]}\n"
        f"  • Pedestrian crossings: {row.iloc[22]}\n"
        f"  • Unclassified: {row.iloc[18]}\n"
        "\n"
        f"- Bus ridership (on/off): {row.iloc[26]} / {row.iloc[27]}\n"
        f"- Subway ridership (on/off): {row.iloc[28]} / {row.iloc[29]}\n"
    )

    road_prompt.append(prompt)

    if (idx + 1) % 5000 == 0:
        print(f"  Generated {idx + 1}/{len(road)} prompts...")

road_prompt_df['prompt'] = road_prompt
print(f"\n[OK] Generated {len(road_prompt_df)} prompts")
print(f"[OK] Shape: {road_prompt_df.shape}")

# Save to CSV
os.makedirs('../dong_prompts', exist_ok=True)
output_path = '../dong_prompts/road_prompts.csv'

print(f"\nSaving prompts to: {output_path}")
road_prompt_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"[OK] Saved successfully")

print("\n" + "="*80)
print("ROAD NETWORK PROMPT GENERATION COMPLETE!")
print("="*80)
print(f"\nOutput file: {output_path} ({len(road_prompt_df)} prompts)")
print(f"Next step: Generate LLM embeddings using this prompt file")
