import re

def format_date(date_str):
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%B %Y")

def parse_group_info(text):
    key_match = re.search(r"key: \('([^']+)', '([^']+)'\)", text)
    location, date = key_match.groups()
    total_rows = int(re.search(r"Total number of rows in this group: (\d+)", text).group(1))
    
    # Parse cluster attributes
    cluster_data = {}
    current_column = None
    
    for line in text.split('\n'):
        if "Column '" in line:
            column_match = re.search(r"Column '([^']+)':", line)
            if column_match:
                current_column = column_match.group(1)
                if "No information" not in line:
                    total_match = re.search(r"Total rows with data: (\d+)", line)
                    if total_match:
                        cluster_data[current_column] = {
                            'total_rows': int(total_match.group(1)),
                            'values': {}
                        }
        elif current_column and current_column in cluster_data:
            value_match = re.match(r'\s+([^:]+):\s*(\d+)', line)
            if value_match:
                value, count = value_match.groups()
                if not ('{' in value or '[' in value):
                    cluster_data[current_column]['values'][value.strip()] = int(count)
    
    # Parse binary attributes
    binary_data = {}
    binary_section = text[text.find("Binary Attributes:"):text.find("Numerical Attributes:")]
    for line in binary_section.split('\n'):
        match = re.search(r"Column '([^']+)': Total rows with data: (\d+), ([\d.]+)% True", line)
        if match and "No information" not in line:
            column_name, total_rows, percentage = match.groups()
            binary_data[column_name] = {
                'total_rows': int(total_rows),
                'percentage': float(percentage)
            }
    
    # Parse numerical attributes
    numerical_data = {}
    current_column = None
    numerical_section = text[text.find("Numerical Attributes:"):]
    
    for line in numerical_section.split('\n'):
        if "Column '" in line and "No information" not in line:
            column_match = re.search(r"Column '([^']+)':", line)
            if column_match:
                current_column = column_match.group(1)
                numerical_data[current_column] = {}
        elif current_column and ':' in line:
            stat, value = [x.strip() for x in line.split(':', 1)]
            if stat in ['Mean', 'Median', 'Mode', 'Min', 'Max', 'Std Dev']:
                numerical_data[current_column][stat] = float(value)
    
    return {
        'location': location,
        'date': date,
        'total_rows': total_rows,
        'cluster_data': cluster_data,
        'binary_data': binary_data,
        'numerical_data': numerical_data
    }

def generate_report(data):
    formatted_date = format_date(data['date'])
    report = f"The following report gives some information about attributes and characteristics of Airbnb listings in {data['location']} for {formatted_date}.\n\n"
    report += f"There are a total of {data['total_rows']} AirBnB listing recorded.\n\n"
    
    # Process cluster attributes (excluding binary attributes)
    cluster_mapping = {
        'Listing Type': ('offer', 'listing type'),
        'Property Type': ('are', 'Properties Type'),
        'Cancellation Policy': ('have', 'Cancelation Policy'),
        'Check-in Time': ('allow check-in', 'Check-in Time'),
        'Checkout Time': ('allow check-out', 'Checkout Time'),
        'Amenities': ('support', 'Amenities'),
        'Airbnb Response Time (Text)': ('response', 'Airbnb Response Time (Text)')
    }
    
    # Skip binary attributes when processing clusters
    binary_columns = set(data['binary_data'].keys())
    
    for column, info in data['cluster_data'].items():
        if column not in data['numerical_data'] and column not in binary_columns:
            verb, display_name = cluster_mapping.get(column, ('have', column))
            report += f"For {display_name}, there are {info['total_rows']} AirBnB listings notifying about their {display_name.lower()}. Among them, "
            
            values = []
            for value, count in info['values'].items():
                if column == 'Cancellation Policy':
                    values.append(f"{count} AirBnB listings {verb} {value} policy")
                else:
                    values.append(f"{count} Airbnb listings {verb} {value}")
            report += ", ".join(values) + ".\n\n"
    
    # Process binary attributes
    for column, info in data['binary_data'].items():
        report += f"For {column}, there are {info['total_rows']} AirBnB listings notifying about their {column}. "
        report += f"Among them, {info['percentage']:.2f}% of AirBnB listings offer {column}.\n\n"
    
    # Process numerical attributes
    for column, stats in data['numerical_data'].items():
        report += f"For {column}, there are {data['total_rows']} AirBnB listings notifying about their {column}. "
        report += f"The mean number of {column} is {stats['Mean']:.2f}. "
        report += f"Meanwhile the median number of {column} is {stats['Median']:.2f}. "
        report += f"The mode number of {column} is {stats['Mode']:.1f}. "
        report += f"Furthermore, the min number of {column} is {stats['Min']:.2f}, "
        report += f"while the max number of {column} is {stats['Max']:.2f}. "
        report += f"The standard deviation is {stats['Std Dev']:.2f}.\n\n"
    
    report += f"Assume you are a data analyst with high experience in analyzing AirBnB market. "
    report += f"Give some comments for the AirBnB activities of {data['location']} in {formatted_date}."
    
    return report

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    groups = content.split("--------------------------------------------------")
    reports = []
    
    for group in groups:
        if group.strip():
            data = parse_group_info(group)
            report = generate_report(data)
            reports.append(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n--------------------------------------------------\n".join(reports))

# Example usage
process_file('quang/llm_embeddings/prompts/new_prompts/raw_prompts_new.txt', 'quang/llm_embeddings/prompts/new_prompts/refined_prompts_new.txt')