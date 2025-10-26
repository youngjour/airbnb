"""
Run Feature Selection Experiments Systematically

Tests different SGIS feature combinations to find optimal subset
that improves upon raw-only baseline (RMSE: 0.810).
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Common training parameters
COMMON_PARAMS = {
    'model': 'transformer',
    'epochs': 5,
    'batch_size': 8,
    'window_size': 9,
    'mode': '3m',
    'label': 'all'
}

# Experiment configurations
EXPERIMENTS = [
    # Phase 1: Critical tests
    {
        'name': 'SGIS-only',
        'description': 'Test standalone SGIS value (6 features)',
        'embed1': 'sgis_improved',
        'embed2': None,
        'expected_features': 6
    },
    {
        'name': 'Raw + Penetration',
        'description': 'Market saturation only (1 feature)',
        'embed1': 'raw',
        'embed2': 'sgis_penetration',
        'expected_features': 1
    },
    {
        'name': 'Raw + No Redundancy',
        'description': 'Remove overlapping features (4 features)',
        'embed1': 'raw',
        'embed2': 'sgis_no_redundancy',
        'expected_features': 4
    },

    # Phase 2: Category tests
    {
        'name': 'Raw + Competition',
        'description': 'Competition & saturation (2 features)',
        'embed1': 'raw',
        'embed2': 'sgis_competition',
        'expected_features': 2
    },
    {
        'name': 'Raw + Attractiveness',
        'description': 'Tourist attractiveness (2 features)',
        'embed1': 'raw',
        'embed2': 'sgis_attractiveness',
        'expected_features': 2
    },
    {
        'name': 'Raw + Ratios',
        'description': 'Business mix ratios (3 features)',
        'embed1': 'raw',
        'embed2': 'sgis_ratios',
        'expected_features': 3
    },

    # Reference (already completed)
    {
        'name': 'Raw + All SGIS',
        'description': 'Current result (6 features)',
        'embed1': 'raw',
        'embed2': 'sgis_improved',
        'expected_features': 6,
        'completed': True,
        'rmse': 1.097
    }
]

# Baseline references
BASELINES = {
    'Raw-only': 0.810,
    'Old SGIS': 1.301,
    'Raw + All SGIS': 1.097
}


def build_command(exp_config):
    """Build training command for experiment."""
    cmd = ['python', 'main.py']

    # Add embeddings
    cmd.extend(['--embed1', exp_config['embed1']])
    if exp_config['embed2']:
        cmd.extend(['--embed2', exp_config['embed2']])

    # Add common parameters
    for key, value in COMMON_PARAMS.items():
        cmd.extend([f'--{key}', str(value)])

    return cmd


def extract_results(output_dir):
    """Extract RMSE from results.json."""
    try:
        results_file = Path(output_dir) / 'results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)

        overall_rmse = results['overall_real_metrics']['overall_avg_RMSE']
        return overall_rmse
    except Exception as e:
        print(f"  Error extracting results: {e}")
        return None


def run_experiment(exp_config, exp_num, total):
    """Run single experiment."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_num}/{total}: {exp_config['name']}")
    print("=" * 80)
    print(f"Description: {exp_config['description']}")
    print(f"Features: {exp_config['expected_features']}")
    print(f"Config: embed1={exp_config['embed1']}, embed2={exp_config['embed2']}")

    # Check if already completed
    if exp_config.get('completed'):
        print(f"\n✓ Already completed - RMSE: {exp_config['rmse']:.3f}")
        return exp_config['rmse']

    # Build and run command
    cmd = build_command(exp_config)
    print(f"\nRunning: {' '.join(cmd)}")
    print("\nTraining...")

    try:
        result = subprocess.run(
            cmd,
            cwd='.',
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes
        )

        if result.returncode != 0:
            print(f"\n✗ Training failed!")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars
            return None

        # Find output directory from stdout
        output_dir = None
        for line in result.stderr.split('\n'):
            if 'Results saved to' in line:
                output_dir = line.split('Results saved to')[-1].strip()
                break

        if not output_dir:
            print(f"\n✗ Could not find output directory")
            return None

        # Extract results
        rmse = extract_results(output_dir)
        if rmse:
            print(f"\n✓ Training completed - RMSE: {rmse:.3f}")

            # Compare to baselines
            raw_baseline = BASELINES['Raw-only']
            improvement = ((raw_baseline - rmse) / raw_baseline) * 100

            if rmse < raw_baseline:
                print(f"  🎉 BEATS RAW-ONLY by {improvement:.1f}%!")
            else:
                print(f"  Worse than raw-only by {-improvement:.1f}%")

            return rmse
        else:
            print(f"\n✗ Could not extract results")
            return None

    except subprocess.TimeoutExpired:
        print(f"\n✗ Training timed out (>15 minutes)")
        return None
    except Exception as e:
        print(f"\n✗ Error running experiment: {e}")
        return None


def create_summary_table(results):
    """Create results summary table."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Add baselines
    summary_data = []
    for name, rmse in BASELINES.items():
        summary_data.append({
            'Experiment': name,
            'Features': '-',
            'RMSE': rmse,
            'vs Raw-only': 0.0 if name == 'Raw-only' else ((rmse - BASELINES['Raw-only']) / BASELINES['Raw-only'] * 100),
            'Status': 'Baseline'
        })

    # Add experiment results
    for exp, rmse in results.items():
        if rmse is None:
            continue

        # Find feature count
        exp_config = next(e for e in EXPERIMENTS if e['name'] == exp)
        features = exp_config['expected_features']

        improvement = ((BASELINES['Raw-only'] - rmse) / BASELINES['Raw-only']) * 100

        status = '✓ BEATS BASELINE' if rmse < BASELINES['Raw-only'] else 'Worse'

        summary_data.append({
            'Experiment': exp,
            'Features': features,
            'RMSE': rmse,
            'vs Raw-only': improvement,
            'Status': status
        })

    # Create DataFrame and display
    df = pd.DataFrame(summary_data)
    df = df.sort_values('RMSE')

    print(f"\n{df.to_string(index=False)}")

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'feature_selection_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")

    # Find best experiment
    best_exp = df[df['Status'] == '✓ BEATS BASELINE']
    if not best_exp.empty:
        best = best_exp.iloc[0]
        print(f"\n🏆 BEST RESULT: {best['Experiment']}")
        print(f"   RMSE: {best['RMSE']:.3f} ({best['Features']} features)")
        print(f"   Improvement: {best['vs Raw-only']:.1f}% vs raw-only")
    else:
        print(f"\n⚠️  No experiment beat the raw-only baseline")


def main():
    """Run all feature selection experiments."""
    print("=" * 80)
    print("SGIS FEATURE SELECTION EXPERIMENTS")
    print("=" * 80)
    print(f"\nTarget: Beat raw-only baseline (RMSE: {BASELINES['Raw-only']:.3f})")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    # Ask for confirmation
    print(f"\nThis will run {len([e for e in EXPERIMENTS if not e.get('completed')])} new experiments")
    print("Estimated time: ~10-15 minutes per experiment")
    response = input("\nProceed? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run experiments
    results = {}
    for i, exp in enumerate(EXPERIMENTS, 1):
        rmse = run_experiment(exp, i, len(EXPERIMENTS))
        results[exp['name']] = rmse

    # Create summary
    create_summary_table(results)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
