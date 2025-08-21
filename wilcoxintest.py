import duckdb
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# base directory for the DuckDB database
BASE_DIR = Path(__file__).parent
DUCKDB_PATH = BASE_DIR / "EXOMISER_VS_LLMS_RAG_Benchmark.duckdb"
conn = duckdb.connect(str(DUCKDB_PATH))

def calculate_reciprocal_ranks(ranks):
    """Convert ranks to reciprocal ranks (1/rank, 0 for rank 0)"""
    return [1.0/rank if rank > 0 else 0.0 for rank in ranks]

def perform_wilcoxon_tests(data_dict, test_type='signed_rank'):
    """Perform pairwise Wilcoxon tests between all models"""
    models = list(data_dict.keys())
    results = []
    
    for model1, model2 in combinations(models, 2):
        data1 = np.array(data_dict[model1])
        data2 = np.array(data_dict[model2])
        
        # Ensure same length (should be from comparison tables)
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        if test_type == 'signed_rank':
            # Wilcoxon signed-rank test (paired samples)
            # Tests if the median difference is zero
            statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
            test_name = "Wilcoxon Signed-Rank"
        else:
            # Mann-Whitney U test (independent samples)
            # Tests if distributions are the same
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        # Calculate effect size (difference in medians)
        median_diff = np.median(data1) - np.median(data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        
        # Determine significance level
        if p_value < 0.001:
            sig_level = "***"
        elif p_value < 0.01:
            sig_level = "**"
        elif p_value < 0.05:
            sig_level = "*"
        else:
            sig_level = "ns"
        
        results.append({
            'model1': model1,
            'model2': model2,
            'test_type': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significance': sig_level,
            'median_diff': median_diff,
            'mean_diff': mean_diff,
            'n_cases': min_len
        })
    
    return results

try:
    print("WILCOXON STATISTICAL ANALYSIS OF LLM PERFORMANCE")
    print("=" * 80)
    
    # Get all comparison tables
    tables = conn.execute("SHOW TABLES").fetchall()
    comparison_tables = [table[0] for table in tables if '_vs_' in table[0] and 'rank_changes' in table[0]]
    
    # Extract ranks for each model from comparison tables
    all_model_ranks = {}
    all_reciprocal_ranks = {}
    
    print("Extracting rank data from comparison tables...")
    for table_name in comparison_tables:
        print(f"Processing: {table_name}")
        
        # Get data from the table
        df = conn.execute(f'SELECT * FROM "{table_name}"').df()
        
        # Extract model names
        models = table_name.replace('_disease_rank_changes', '').split('_vs_')
        model1, model2 = models[0], models[1]
        
        # Get rank columns
        model1_col = df.columns[2]
        model2_col = df.columns[3]
        
        model1_ranks = df[model1_col].tolist()
        model2_ranks = df[model2_col].tolist()
        
        # Store ranks
        if model1 not in all_model_ranks:
            all_model_ranks[model1] = []
        if model2 not in all_model_ranks:
            all_model_ranks[model2] = []
            
        all_model_ranks[model1].extend(model1_ranks)
        all_model_ranks[model2].extend(model2_ranks)
    
    # Calculate reciprocal ranks for each model
    for model, ranks in all_model_ranks.items():
        all_reciprocal_ranks[model] = calculate_reciprocal_ranks(ranks)
    
    print(f"\nCollected rank data for {len(all_model_ranks)} models")
    for model, ranks in all_model_ranks.items():
        print(f"  {model}: {len(ranks)} cases")
    
    # Perform pairwise Wilcoxon tests on reciprocal ranks
    print("\n" + "=" * 80)
    print("PAIRWISE WILCOXON SIGNED-RANK TESTS (RECIPROCAL RANKS)")
    print("=" * 80)
    print("Testing: H0: Median difference in reciprocal ranks = 0")
    print("Higher reciprocal rank = better performance")
    print()
    
    # For signed-rank test, we need paired data from the same test cases
    # Let's extract this from the comparison tables directly
    pairwise_results = []
    
    for table_name in comparison_tables:
        df = conn.execute(f'SELECT * FROM "{table_name}"').df()
        models = table_name.replace('_disease_rank_changes', '').split('_vs_')
        model1, model2 = models[0], models[1]
        
        # Get ranks for the same test cases
        ranks1 = df[df.columns[2]].tolist()
        ranks2 = df[df.columns[3]].tolist()
        
        # Convert to reciprocal ranks
        rr1 = calculate_reciprocal_ranks(ranks1)
        rr2 = calculate_reciprocal_ranks(ranks2)
        
        # Perform paired Wilcoxon test
        statistic, p_value = wilcoxon(rr1, rr2, alternative='two-sided')
        
        median_diff = np.median(rr1) - np.median(rr2)
        mean_diff = np.mean(rr1) - np.mean(rr2)
        
        # Effect size (Cohen's d equivalent for ranks)
        pooled_std = np.sqrt((np.var(rr1) + np.var(rr2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Significance level
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        pairwise_results.append({
            'comparison': f"{model1} vs {model2}",
            'model1': model1,
            'model2': model2,
            'statistic': statistic,
            'p_value': p_value,
            'significance': sig,
            'median_diff': median_diff,
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'n_cases': len(rr1),
            'model1_median_rr': np.median(rr1),
            'model2_median_rr': np.median(rr2),
            'better_model': model1 if median_diff > 0 else model2
        })
    
    # Display results
    print(f"{'Comparison':<35} {'Statistic':<10} {'p-value':<10} {'Sig':<4} {'Effect Size':<10} {'Better Model':<15}")
    print("-" * 95)
    
    for result in sorted(pairwise_results, key=lambda x: x['p_value']):
        print(f"{result['comparison']:<35} {result['statistic']:<10.1f} {result['p_value']:<10.6f} "
              f"{result['significance']:<4} {result['cohens_d']:<10.3f} {result['better_model']:<15}")
    
    # Summary of significant differences
    print("\n" + "=" * 80)
    print("SUMMARY OF SIGNIFICANT DIFFERENCES")
    print("=" * 80)
    
    significant_results = [r for r in pairwise_results if r['p_value'] < 0.05]
    
    if significant_results:
        print(f"Found {len(significant_results)} significant differences (p < 0.05):")
        for result in sorted(significant_results, key=lambda x: x['p_value']):
            direction = "significantly better" if result['median_diff'] > 0 else "significantly worse"
            print(f"• {result['model1']} is {direction} than {result['model2']} "
                  f"(p = {result['p_value']:.6f}, effect size = {result['cohens_d']:.3f})")
    else:
        print("No statistically significant differences found between any model pairs.")
    
    # Create summary statistics
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS BY MODEL")
    print("=" * 80)
    
    stats_data = []
    for model, rr_values in all_reciprocal_ranks.items():
        stats_data.append({
            'Model': model,
            'N Cases': len(rr_values),
            'Mean RR': np.mean(rr_values),
            'Median RR': np.median(rr_values),
            'Std RR': np.std(rr_values),
            'Q1 RR': np.percentile(rr_values, 25),
            'Q3 RR': np.percentile(rr_values, 75),
            'Min RR': np.min(rr_values),
            'Max RR': np.max(rr_values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.round(4).to_string(index=False))
    
    # Export results
    results_df = pd.DataFrame(pairwise_results)
    results_df.to_csv('wilcoxon_test_results_all_LLMs_RAG_VS_EXOMISER.csv', index=False)
    
    stats_df.to_csv('model_descriptive_stats_all_LLMs_RAG_VS_EXOMISER.csv', index=False)
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("• Reciprocal Rank: 1/rank (higher = better)")
    print("• Wilcoxon Signed-Rank Test: Tests if median difference ≠ 0")
    print("• p < 0.05: Statistically significant difference")
    print("• Effect Size (Cohen's d): 0.2=small, 0.5=medium, 0.8=large")
    print("• Positive median difference: Model1 > Model2")
    print("• Files saved: 'wilcoxon_test_results.csv', 'model_descriptive_stats.csv'")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()