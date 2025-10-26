"""
Generate Human Flow LLM prompts from Human_flow.csv

Based on Hongju's make_additional_llm_prompt.ipynb
"""

import pandas as pd
import os

print("="*80)
print("HUMAN FLOW PROMPT GENERATION")
print("="*80)

# Load human flow data
file_path = '../../../Data/Preprocessed_data/Dong/Human_flow.csv'
print(f"\nLoading human flow data from: {file_path}")
hf = pd.read_csv(file_path)
print(f"[OK] Loaded {len(hf)} rows x {len(hf.columns)} columns")

# Create prompt dataframe
hf_prompt_df = hf[['Reporting Month', 'Dong_name']].copy()
hf_prompt = []

print("\nGenerating human flow prompts...")

def get_mean(a, b):
    """Calculate mean of two values"""
    return (a + b) / 2

for idx, row in hf.iterrows():
    month = row['Reporting Month']
    dong = row['Dong_name']

    prompt = (
        f"[{month} | {dong}] Average Floating Population Summary:\n"
        f"- Domestic Floating Population Summary\n"
        f"- Total Domestic Floating Population: {row.iloc[2]:.2f}\n"
        f"- Domestic Floating Population by Age and Gender\n"
        f"- Teens Male: {get_mean(row.iloc[4], row.iloc[5]):.2f}, Female: {get_mean(row.iloc[18], row.iloc[19]):.2f}\n"
        f"- 20s Male: {get_mean(row.iloc[6], row.iloc[7]):.2f}, Female: {get_mean(row.iloc[20], row.iloc[21]):.2f}\n"
        f"- 30s Male: {get_mean(row.iloc[8], row.iloc[9]):.2f}, Female: {get_mean(row.iloc[22], row.iloc[23]):.2f}\n"
        f"- 40s Male: {get_mean(row.iloc[10], row.iloc[11]):.2f}, Female: {get_mean(row.iloc[24], row.iloc[25]):.2f}\n"
        f"- 50s Male: {get_mean(row.iloc[12], row.iloc[13]):.2f}, Female: {get_mean(row.iloc[26], row.iloc[27]):.2f}\n"
        f"- 60s Male: {get_mean(row.iloc[14], row.iloc[15]):.2f}, Female: {get_mean(row.iloc[28], row.iloc[29]):.2f}\n"
        f"- 70 and above Male: {row.iloc[16]}, Female: {row.iloc[30]}\n"
        '\n'
        f"- Long-term Foreign Resident Floating Population Summary\n"
        f"- Total Long-term Foreign Resident Floating Population: {row.iloc[31]:.2f}\n"
        f"- Chinese Long-term Residents: {row.iloc[32]:.2f}\n"
        f"- Non-Chinese Long-term Residents: {row.iloc[33]:.2f}\n"
        '\n'
        f"- Short-term Foreign Visitor Floating Population Summary\n"
        f"- Total Short-term Foreign Visitor Floating Population: {row.iloc[34]:.2f}\n"
        f"- Chinese Short-term Visitors: {row.iloc[35]:.2f}\n"
        f"- Non-Chinese Short-term Visitors: {row.iloc[36]:.2f}\n"
    )

    hf_prompt.append(prompt)

    if (idx + 1) % 5000 == 0:
        print(f"  Generated {idx + 1}/{len(hf)} prompts...")

hf_prompt_df['prompt'] = hf_prompt
print(f"\n[OK] Generated {len(hf_prompt_df)} prompts")
print(f"[OK] Shape: {hf_prompt_df.shape}")

# Save to CSV
os.makedirs('../dong_prompts', exist_ok=True)
output_path = '../dong_prompts/human_flow_prompts.csv'

print(f"\nSaving prompts to: {output_path}")
hf_prompt_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"[OK] Saved successfully")

print("\n" + "="*80)
print("HUMAN FLOW PROMPT GENERATION COMPLETE!")
print("="*80)
print(f"\nOutput file: {output_path} ({len(hf_prompt_df)} prompts)")
print(f"Next step: Generate LLM embeddings using this prompt file")
