"""
Generate Natural Language Prompts for SGIS Local Features
Creates prompts describing local district characteristics for LLM embedding generation.
"""

import pandas as pd
import numpy as np

def create_local_feature_prompt(row):
    """
    Create a natural language prompt describing local district characteristics.

    Features:
    - retail_ratio: Tourism attractiveness (shopping)
    - accommodation_ratio: Competition indicator
    - restaurant_ratio: Tourism attractiveness (dining)
    - housing_units: Market size
    - airbnb_listing_count: Current market presence
    - airbnb_per_1k_housing: Market saturation
    """

    date = row['Reporting Month']
    dong_name = row['Dong_name']

    # Extract feature values
    retail_ratio = row['retail_ratio']
    accommodation_ratio = row['accommodation_ratio']
    restaurant_ratio = row['restaurant_ratio']
    housing_units = int(row['housing_units'])
    airbnb_count = int(row['airbnb_listing_count'])
    airbnb_penetration = row['airbnb_per_1k_housing']

    # Create structured prompt
    prompt = f"[{date} | {dong_name}] Local District Feature Summary:\n\n"

    # Business Composition Section
    prompt += "BUSINESS COMPOSITION:\n"
    prompt += f"This district has {retail_ratio:.2f}% retail businesses, "
    prompt += f"{accommodation_ratio:.2f}% accommodation facilities, "
    prompt += f"and {restaurant_ratio:.2f}% restaurants and food services.\n"

    # Interpretation based on ratios
    if retail_ratio > 25:
        prompt += "The high retail ratio indicates a major shopping destination with strong commercial activity. "
    elif retail_ratio > 20:
        prompt += "The moderate-to-high retail presence suggests good shopping amenities. "
    elif retail_ratio < 10:
        prompt += "The low retail ratio indicates limited commercial shopping options. "
    else:
        prompt += "The retail presence is at typical levels for residential areas. "

    if restaurant_ratio > 25:
        prompt += "The high restaurant ratio indicates a major dining and entertainment district. "
    elif restaurant_ratio > 15:
        prompt += "The healthy restaurant presence suggests a vibrant food scene. "
    elif restaurant_ratio < 5:
        prompt += "The limited restaurant options suggest primarily residential character. "
    else:
        prompt += "Restaurant density is at typical residential levels. "

    if accommodation_ratio > 2:
        prompt += "The high accommodation ratio indicates significant hotel and lodging competition.\n"
    elif accommodation_ratio > 0.5:
        prompt += "There is moderate competition from other accommodation providers.\n"
    elif accommodation_ratio > 0:
        prompt += "Minimal traditional accommodation competition exists.\n"
    else:
        prompt += "No traditional accommodation facilities are present.\n"

    prompt += "\n"

    # Market Size & Saturation Section
    prompt += "MARKET CHARACTERISTICS:\n"
    prompt += f"The district contains {housing_units:,} housing units, representing the potential market size. "

    if housing_units > 15000:
        prompt += "This is a very large residential district. "
    elif housing_units > 8000:
        prompt += "This is a moderately large residential area. "
    elif housing_units > 3000:
        prompt += "This is a medium-sized residential district. "
    elif housing_units > 0:
        prompt += "This is a small residential area. "
    else:
        prompt += "This is a non-residential or mixed-use district. "

    prompt += f"There are currently {airbnb_count} Airbnb listings, "
    prompt += f"resulting in {airbnb_penetration:.2f} listings per 1,000 housing units.\n"

    # Market saturation interpretation
    if airbnb_penetration > 10:
        prompt += "This represents very high Airbnb market penetration and saturation. "
    elif airbnb_penetration > 5:
        prompt += "This represents high Airbnb market penetration. "
    elif airbnb_penetration > 2:
        prompt += "This represents moderate Airbnb market presence. "
    elif airbnb_penetration > 0.5:
        prompt += "This represents low Airbnb market penetration. "
    elif airbnb_penetration > 0:
        prompt += "This represents minimal Airbnb presence. "
    else:
        prompt += "There is no Airbnb presence in this district. "

    prompt += "\n\n"

    # Tourism & Investment Context
    prompt += "TOURISM & INVESTMENT POTENTIAL:\n"

    # Calculate combined attractiveness score (retail + restaurant)
    attractiveness_score = retail_ratio + restaurant_ratio

    if attractiveness_score > 45:
        prompt += "The combined retail and dining infrastructure (total {:.1f}%) suggests this is a major tourism and commercial hub. ".format(attractiveness_score)
    elif attractiveness_score > 35:
        prompt += "The strong retail and dining infrastructure (total {:.1f}%) suggests good tourism potential. ".format(attractiveness_score)
    elif attractiveness_score > 25:
        prompt += "The moderate retail and dining options (total {:.1f}%) provide basic tourism amenities. ".format(attractiveness_score)
    else:
        prompt += "The limited retail and dining infrastructure (total {:.1f}%) suggests primarily residential character. ".format(attractiveness_score)

    # Market opportunity assessment
    if airbnb_penetration < 1 and attractiveness_score > 30:
        prompt += "Despite strong tourism amenities, Airbnb penetration is low, suggesting potential growth opportunity. "
    elif airbnb_penetration > 10 and attractiveness_score > 35:
        prompt += "Both high tourism amenities and high Airbnb saturation indicate a mature, competitive market. "
    elif airbnb_penetration > 5 and attractiveness_score < 20:
        prompt += "High Airbnb penetration despite limited tourism amenities may indicate supply-demand imbalance. "
    elif airbnb_penetration < 2 and attractiveness_score < 20:
        prompt += "Low tourism amenities and minimal Airbnb presence suggest limited short-term rental appeal. "

    if accommodation_ratio > 1 and airbnb_penetration > 3:
        prompt += "Significant competition exists from both traditional accommodations and other Airbnb listings. "

    prompt += "\n\n"

    # Analyst perspective
    prompt += "Based on these local district characteristics, consider the competitive dynamics, "
    prompt += "tourism attractiveness, market saturation, and investment potential for short-term rental properties."

    return prompt

def main():
    print("Loading SGIS improved features...")
    df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    # Generate prompts for all rows
    print("Generating natural language prompts...")
    prompts = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%)")

        prompt_text = create_local_feature_prompt(row)
        prompts.append({
            'Reporting Month': row['Reporting Month'],
            'Dong_name': row['Dong_name'],
            'prompt': prompt_text
        })

    # Create DataFrame
    prompt_df = pd.DataFrame(prompts)

    # Save to CSV
    output_file = 'sgis_local_prompts.csv'
    prompt_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ Generated {len(prompt_df)} prompts")
    print(f"✓ Saved to: {output_file}")
    print(f"\nSample prompt (first entry):")
    print("=" * 80)
    print(prompt_df['prompt'].iloc[0])
    print("=" * 80)

    # Statistics
    print(f"\nPrompt Statistics:")
    print(f"  Average length: {prompt_df['prompt'].str.len().mean():.0f} characters")
    print(f"  Min length: {prompt_df['prompt'].str.len().min()} characters")
    print(f"  Max length: {prompt_df['prompt'].str.len().max()} characters")

if __name__ == "__main__":
    main()
