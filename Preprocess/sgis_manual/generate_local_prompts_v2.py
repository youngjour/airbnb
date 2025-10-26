"""
Generate Improved Natural Language Prompts for SGIS Local Features (v2)
Creates Airbnb-specific prompts with interaction effects and nuanced interpretations.

Key improvements over v1:
1. Airbnb-specific demand driver focus
2. Feature interaction effects
3. Softer, more nuanced language
4. Competitive dynamics emphasis
5. Market opportunity framing
"""

import pandas as pd
import numpy as np

def get_retail_interpretation(retail_ratio, restaurant_ratio):
    """Interpret retail ratio with Airbnb-specific context."""
    combined_commercial = retail_ratio + restaurant_ratio

    if retail_ratio > 25:
        return "This is a major shopping destination with extensive retail infrastructure, likely attracting shopping tourists who may prefer Airbnb for multi-day stays near shopping districts."
    elif retail_ratio > 20:
        return "Strong retail presence suggests good shopping amenities, which can attract tourists seeking authentic local shopping experiences while staying in Airbnb accommodations."
    elif retail_ratio > 15:
        return "Moderate retail infrastructure provides convenient shopping options for Airbnb guests during their stay."
    elif retail_ratio > 10:
        return "Basic retail options are available, sufficient for essential needs of short-term Airbnb guests."
    elif retail_ratio > 5:
        return "Limited retail presence suggests a primarily residential character, which may appeal to Airbnb guests seeking authentic neighborhood experiences away from commercial areas."
    else:
        return "Minimal retail infrastructure indicates a quiet residential area, potentially attractive to Airbnb guests valuing peaceful, local neighborhood stays."

def get_restaurant_interpretation(restaurant_ratio, retail_ratio):
    """Interpret restaurant ratio with Airbnb demand context."""
    if restaurant_ratio > 25:
        return "Exceptional dining density creates a major food and entertainment district, a strong draw for culinary tourists who often prefer Airbnb's longer stays and kitchen facilities for experiencing local food culture."
    elif restaurant_ratio > 20:
        return "Very strong dining scene with abundant restaurant options, attractive to food-focused travelers who value Airbnb's proximity to diverse dining options."
    elif restaurant_ratio > 15:
        return "Vibrant restaurant presence offers varied dining experiences, enhancing the neighborhood's appeal for Airbnb guests seeking authentic local cuisine."
    elif restaurant_ratio > 10:
        return "Healthy restaurant density provides good dining variety for Airbnb guests, supporting multi-day stays."
    elif restaurant_ratio > 5:
        return "Moderate dining options available, adequate for essential meals during Airbnb stays."
    else:
        return "Limited dining infrastructure suggests primarily residential character, where Airbnb guests may rely more on in-unit cooking facilities."

def get_accommodation_competition(accommodation_ratio, airbnb_penetration):
    """Analyze accommodation competition with market implications."""
    if accommodation_ratio > 2:
        return f"High concentration of traditional accommodations (hotels/motels) creates significant competition for Airbnb. However, current Airbnb penetration of {airbnb_penetration:.1f} per 1,000 housing units suggests the market supports both traditional and alternative lodging."
    elif accommodation_ratio > 1:
        return f"Moderate traditional accommodation presence indicates some hotel competition. Airbnb penetration of {airbnb_penetration:.1f} per 1,000 units shows how alternative lodging complements traditional options."
    elif accommodation_ratio > 0.5:
        return f"Low traditional accommodation density means Airbnb faces minimal hotel competition, with {airbnb_penetration:.1f} listings per 1,000 units representing the primary short-term lodging option."
    elif accommodation_ratio > 0:
        return f"Minimal hotel presence leaves short-term lodging market largely to Airbnb, with current penetration of {airbnb_penetration:.1f} per 1,000 units."
    else:
        return f"No traditional accommodations exist in this district, making Airbnb the sole short-term lodging option with {airbnb_penetration:.1f} listings per 1,000 housing units."

def get_market_opportunity(retail_ratio, restaurant_ratio, accommodation_ratio, airbnb_penetration, housing_units):
    """Assess market opportunity with interaction effects."""
    tourism_score = retail_ratio + restaurant_ratio

    # Market size context
    if housing_units > 15000:
        size_desc = f"very large residential market ({housing_units:,} units)"
    elif housing_units > 10000:
        size_desc = f"large residential market ({housing_units:,} units)"
    elif housing_units > 5000:
        size_desc = f"moderately-sized residential market ({housing_units:,} units)"
    elif housing_units > 2000:
        size_desc = f"medium residential market ({housing_units:,} units)"
    else:
        size_desc = f"small residential market ({housing_units:,} units)"

    analysis = f"This district has a {size_desc}. "

    # Opportunity assessment based on interactions
    if tourism_score > 40 and airbnb_penetration < 3:
        analysis += f"STRONG GROWTH OPPORTUNITY: Despite excellent tourism infrastructure (retail + dining = {tourism_score:.1f}%), Airbnb penetration is relatively low ({airbnb_penetration:.1f} per 1,000 units), suggesting significant untapped demand for short-term rentals in this attractive location."
    elif tourism_score > 35 and airbnb_penetration < 5:
        analysis += f"GROWTH POTENTIAL: Strong tourism amenities (retail + dining = {tourism_score:.1f}%) combined with moderate Airbnb penetration ({airbnb_penetration:.1f} per 1,000 units) indicate room for market expansion."
    elif tourism_score > 30 and airbnb_penetration > 10:
        analysis += f"MATURE COMPETITIVE MARKET: High tourism amenities (retail + dining = {tourism_score:.1f}%) and high Airbnb saturation ({airbnb_penetration:.1f} per 1,000 units) indicate a mature, competitive short-term rental market with established demand."
    elif tourism_score < 20 and airbnb_penetration > 8:
        analysis += f"POTENTIAL OVERSUPPLY: Limited tourism infrastructure (retail + dining = {tourism_score:.1f}%) but high Airbnb penetration ({airbnb_penetration:.1f} per 1,000 units) may indicate supply-demand imbalance, possibly serving as residential overflow rather than tourist destination."
    elif tourism_score > 25 and accommodation_ratio > 1.5:
        analysis += f"COMPETITIVE ENVIRONMENT: Good tourism amenities (retail + dining = {tourism_score:.1f}%) but significant traditional accommodation competition ({accommodation_ratio:.2f}%) means Airbnb must differentiate through price, location, or experience. Current penetration: {airbnb_penetration:.1f} per 1,000 units."
    elif tourism_score > 30:
        analysis += f"TOURISM DESTINATION: Strong tourism infrastructure (retail + dining = {tourism_score:.1f}%) supports consistent short-term rental demand. Current Airbnb penetration: {airbnb_penetration:.1f} per 1,000 units."
    elif airbnb_penetration > 5:
        analysis += f"ESTABLISHED AIRBNB MARKET: Moderate Airbnb presence ({airbnb_penetration:.1f} per 1,000 units) despite limited tourism amenities (retail + dining = {tourism_score:.1f}%) suggests serving business travelers, long-term visitors, or local event attendees."
    else:
        analysis += f"RESIDENTIAL CHARACTER: Limited commercial amenities (retail + dining = {tourism_score:.1f}%) and low Airbnb penetration ({airbnb_penetration:.1f} per 1,000 units) indicate primarily residential area with minimal short-term rental appeal."

    return analysis

def get_demand_drivers(retail_ratio, restaurant_ratio, accommodation_ratio, airbnb_penetration):
    """Identify key Airbnb demand drivers for this district."""
    tourism_score = retail_ratio + restaurant_ratio

    drivers = []

    # Primary demand driver
    if restaurant_ratio > 20:
        drivers.append("culinary tourism (dining-focused travelers)")
    if retail_ratio > 20:
        drivers.append("shopping tourism")
    if tourism_score > 35:
        drivers.append("general tourism and leisure")
    if tourism_score < 20 and airbnb_penetration > 3:
        drivers.append("business travel or local events")
    if accommodation_ratio < 0.5 and tourism_score > 15:
        drivers.append("underserved lodging market")

    if not drivers:
        drivers.append("residential overflow or niche demand")

    return "Primary Airbnb demand drivers: " + ", ".join(drivers) + "."

def create_improved_local_prompt(row):
    """
    Create enhanced natural language prompt with Airbnb-specific insights.
    """
    date = row['Reporting Month']
    dong_name = row['Dong_name']

    # Extract features
    retail_ratio = row['retail_ratio']
    accommodation_ratio = row['accommodation_ratio']
    restaurant_ratio = row['restaurant_ratio']
    housing_units = int(row['housing_units'])
    airbnb_count = int(row['airbnb_listing_count'])
    airbnb_penetration = row['airbnb_per_1k_housing']

    # Build enhanced prompt
    prompt = f"[{date} | {dong_name}] Airbnb Market Analysis - Local District Characteristics:\n\n"

    # Section 1: Business Infrastructure & Tourism Appeal
    prompt += "TOURISM & COMMERCIAL INFRASTRUCTURE:\n"
    prompt += f"Business composition: {retail_ratio:.2f}% retail, {restaurant_ratio:.2f}% restaurants, {accommodation_ratio:.2f}% traditional accommodations.\n\n"

    prompt += get_retail_interpretation(retail_ratio, restaurant_ratio) + " "
    prompt += get_restaurant_interpretation(restaurant_ratio, retail_ratio) + "\n\n"

    # Section 2: Competitive Landscape
    prompt += "COMPETITIVE DYNAMICS:\n"
    prompt += get_accommodation_competition(accommodation_ratio, airbnb_penetration) + "\n\n"

    # Section 3: Market Size & Saturation
    prompt += "MARKET CHARACTERISTICS:\n"
    prompt += get_market_opportunity(retail_ratio, restaurant_ratio, accommodation_ratio,
                                     airbnb_penetration, housing_units) + "\n\n"

    # Section 4: Demand Drivers
    prompt += "DEMAND ASSESSMENT:\n"
    prompt += get_demand_drivers(retail_ratio, restaurant_ratio, accommodation_ratio,
                                 airbnb_penetration) + " "

    # Current market state
    if airbnb_count > 0:
        prompt += f"Current market shows {airbnb_count} active Airbnb listings serving this {housing_units:,}-unit residential district.\n\n"
    else:
        prompt += f"No active Airbnb listings currently in this {housing_units:,}-unit residential district.\n\n"

    # Section 5: Investment Perspective
    prompt += "INVESTMENT IMPLICATIONS:\n"
    tourism_score = retail_ratio + restaurant_ratio

    if tourism_score > 35 and airbnb_penetration < 5:
        prompt += "HIGH OPPORTUNITY: Strong fundamentals with room for growth. Low competition risk."
    elif tourism_score > 30 and airbnb_penetration < 8:
        prompt += "MODERATE OPPORTUNITY: Good tourism appeal with manageable competition levels."
    elif tourism_score > 25 and airbnb_penetration > 10:
        prompt += "COMPETITIVE MARKET: Established demand but high saturation requires differentiation."
    elif tourism_score > 25 and accommodation_ratio > 1.5:
        prompt += "COMPETITIVE PRESSURE: Must compete with both traditional lodging and other Airbnbs."
    elif tourism_score < 20 and airbnb_penetration > 8:
        prompt += "RISK CONSIDERATION: High saturation relative to tourism amenities may indicate oversupply."
    elif tourism_score < 20:
        prompt += "LIMITED APPEAL: Residential character with minimal tourism infrastructure."
    else:
        prompt += "NICHE MARKET: Specific demand drivers (events, business) rather than general tourism."

    return prompt

def main():
    print("Loading SGIS improved features...")
    df = pd.read_csv('sgis_improved_final.csv', encoding='utf-8-sig')

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    print("Generating improved Airbnb-specific prompts...")
    prompts = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%)")

        prompt_text = create_improved_local_prompt(row)
        prompts.append({
            'Reporting Month': row['Reporting Month'],
            'Dong_name': row['Dong_name'],
            'prompt': prompt_text
        })

    # Create DataFrame
    prompt_df = pd.DataFrame(prompts)

    # Save to CSV
    output_file = 'sgis_local_prompts_v2.csv'
    prompt_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n[OK] Generated {len(prompt_df)} improved prompts")
    print(f"[OK] Saved to: {output_file}")
    print(f"\nSample improved prompt (first entry):")
    print("=" * 100)
    print(prompt_df['prompt'].iloc[0])
    print("=" * 100)

    # Statistics
    print(f"\nPrompt Statistics:")
    print(f"  Average length: {prompt_df['prompt'].str.len().mean():.0f} characters")
    print(f"  Min length: {prompt_df['prompt'].str.len().min()} characters")
    print(f"  Max length: {prompt_df['prompt'].str.len().max()} characters")

    # Show v1 vs v2 comparison
    print(f"\n" + "=" * 100)
    print("COMPARISON: v1 vs v2 prompt for same district")
    print("=" * 100)

    # Load v1 for comparison
    try:
        v1_df = pd.read_csv('sgis_local_prompts.csv', encoding='utf-8-sig')
        print("\nV1 PROMPT:")
        print("-" * 100)
        print(v1_df['prompt'].iloc[0])
        print("\nV2 PROMPT (IMPROVED):")
        print("-" * 100)
        print(prompt_df['prompt'].iloc[0])
    except:
        print("(Could not load v1 prompts for comparison)")

if __name__ == "__main__":
    main()
