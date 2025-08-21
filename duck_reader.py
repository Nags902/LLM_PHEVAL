# # import duckdb
# # from pathlib import Path
# # import pandas as pd

# # # base directory for the DuckDB database
# # BASE_DIR = Path(__file__).parent
# # DUCKDB_PATH = BASE_DIR / "LLMs_NO_RAG_Benchmark.duckdb"
# # conn = duckdb.connect(str(DUCKDB_PATH))

# # def calculate_top_k_accuracy(ranks, k):
# #     """Calculate top-k accuracy: percentage of cases where correct answer is in top k ranks"""
# #     # Rank 0 typically means the model didn't find the correct answer at all
# #     # Ranks 1-k mean the correct answer was found within top k
# #     successful_cases = sum(1 for rank in ranks if 1 <= rank <= k)
# #     total_cases = len(ranks)
# #     return (successful_cases / total_cases) * 100 if total_cases > 0 else 0

# # try:
# #     # Get all comparison tables
# #     tables = conn.execute("SHOW TABLES").fetchall()
# #     comparison_tables = [table[0] for table in tables if '_vs_' in table[0] and 'rank_changes' in table[0]]
    
# #     print("LLM TOP-K ACCURACY ANALYSIS")
# #     print("=" * 70)
# #     print("NOTE: Rank 0 = model failed to find correct diagnosis")
# #     print("      Rank 1 = correct diagnosis was model's top choice")
# #     print("      Rank n = correct diagnosis was model's nth choice")
# #     print("=" * 70)
    
# #     # Dictionary to store all ranks for each model across all comparisons
# #     all_model_ranks = {}
    
# #     # Analyze each comparison table
# #     for table_name in comparison_tables:
# #         print(f"\nTable: {table_name}")
# #         print("-" * 50)
        
# #         # Get data from the table
# #         df = conn.execute(f'SELECT * FROM "{table_name}"').df()
        
# #         # Extract model names from table name
# #         models = table_name.replace('_disease_rank_changes', '').split('_vs_')
# #         model1, model2 = models[0], models[1]
        
# #         # Get the rank columns (should be columns 2 and 3)
# #         model1_col = df.columns[2]  # First model's ranks
# #         model2_col = df.columns[3]  # Second model's ranks
        
# #         model1_ranks = df[model1_col].tolist()
# #         model2_ranks = df[model2_col].tolist()
        
# #         total_cases = len(df)
        
# #         print(f"Total test cases: {total_cases}")
# #         print()
        
# #         # Calculate top-k accuracies for this comparison
# #         k_values = [1, 3, 5, 10]
        
# #         print(f"{'Model':<25} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<8}")
# #         print("-" * 65)
        
# #         for model, ranks in [(model1, model1_ranks), (model2, model2_ranks)]:
# #             accuracies = []
# #             for k in k_values:
# #                 accuracy = calculate_top_k_accuracy(ranks, k)
# #                 accuracies.append(f"{accuracy:.1f}%")
            
# #             print(f"{model:<25} {accuracies[0]:<8} {accuracies[1]:<8} {accuracies[2]:<8} {accuracies[3]:<8}")
            
# #             # Store ranks for overall analysis
# #             if model not in all_model_ranks:
# #                 all_model_ranks[model] = []
# #             all_model_ranks[model].extend(ranks)
        
# #         # Show rank distributions
# #         print(f"\nRank distribution for {model1}:")
# #         rank_dist1 = pd.Series(model1_ranks).value_counts().sort_index()
# #         for rank, count in rank_dist1.items():
# #             print(f"  Rank {rank}: {count} cases ({count/total_cases*100:.1f}%)")
        
# #         print(f"\nRank distribution for {model2}:")
# #         rank_dist2 = pd.Series(model2_ranks).value_counts().sort_index()
# #         for rank, count in rank_dist2.items():
# #             print(f"  Rank {rank}: {count} cases ({count/total_cases*100:.1f}%)")
    
# #     # Overall analysis across all comparisons
# #     print("\n" + "=" * 70)
# #     print("OVERALL TOP-K ACCURACY ACROSS ALL TEST CASES")
# #     print("=" * 70)
    
# #     # Calculate total cases for each model
# #     model_totals = {model: len(ranks) for model, ranks in all_model_ranks.items()}
    
# #     print(f"{'Model':<25} {'Total Cases':<12} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<8}")
# #     print("-" * 75)
    
# #     for model in sorted(all_model_ranks.keys()):
# #         ranks = all_model_ranks[model]
# #         total = len(ranks)
        
# #         accuracies = []
# #         for k in [1, 3, 5, 10]:
# #             accuracy = calculate_top_k_accuracy(ranks, k)
# #             accuracies.append(f"{accuracy:.1f}%")
        
# #         print(f"{model:<25} {total:<12} {accuracies[0]:<8} {accuracies[1]:<8} {accuracies[2]:<8} {accuracies[3]:<8}")
    
# #     # Additional detailed breakdown
# #     print("\n" + "=" * 70)
# #     print("DETAILED BREAKDOWN")
# #     print("=" * 70)
    
# #     for model in sorted(all_model_ranks.keys()):
# #         ranks = all_model_ranks[model]
# #         total = len(ranks)
        
# #         print(f"\n{model} (Total: {total} cases):")
# #         print("-" * 40)
        
# #         # Count how many got rank 0 (complete failures)
# #         failures = sum(1 for rank in ranks if rank == 0)
# #         print(f"  Failed to find diagnosis (Rank 0): {failures} ({failures/total*100:.1f}%)")
        
# #         # Show success breakdown
# #         for k in [1, 3, 5, 10]:
# #             successful = sum(1 for rank in ranks if 1 <= rank <= k)
# #             print(f"  Found in top {k}: {successful} ({successful/total*100:.1f}%)")
        
# #         # Show rank distribution summary
# #         rank_counts = pd.Series(ranks).value_counts().sort_index()
# #         print(f"  Rank distribution: {dict(rank_counts)}")

# # except Exception as e:
# #     print(f"Error: {e}")
# # finally:
# #     conn.close()

# # import duckdb
# # from pathlib import Path

# # # base directory for the DuckDB database
# # BASE_DIR = Path(__file__).parent
# # DUCKDB_PATH = BASE_DIR / "LLMs_NO_RAG_BENCHMARK/LLMs_NO_RAG_Benchmark.duckdb"
# # conn = duckdb.connect(str(DUCKDB_PATH))

# # try:
# #     print("CHECKING WHAT'S IN THE DUCKDB DATABASE")
# #     print("=" * 60)
    
# #     # Get all tables
# #     tables = conn.execute("SHOW TABLES").fetchall()
# #     print("Available tables:")
# #     for table in tables:
# #         print(f"  - {table[0]}")
    
# #     # Check the summary table structure (this usually contains MRR)
# #     summary_table = "LLMs_NO_RAG_Benchmark_disease_summary"  # Adjust name if different
    
# #     print(f"\n=== STRUCTURE OF {summary_table} ===")
# #     try:
# #         # Get column names
# #         columns = conn.execute(f'DESCRIBE "{summary_table}"').fetchall()
# #         print("Columns in summary table:")
# #         for col in columns:
# #             print(f"  {col[0]}: {col[1]}")
        
# #         # Show sample data
# #         print(f"\n=== SAMPLE DATA FROM {summary_table} ===")
# #         sample = conn.execute(f'SELECT * FROM "{summary_table}" LIMIT 3').fetchall()
# #         col_names = [col[0] for col in columns]
        
# #         for i, row in enumerate(sample):
# #             print(f"\nRow {i+1}:")
# #             for j, value in enumerate(row):
# #                 print(f"  {col_names[j]}: {value}")
                
# #     except Exception as e:
# #         print(f"Could not access summary table: {e}")
        
# #         # Try to find tables with "summary" in the name
# #         print("\nLooking for summary-like tables:")
# #         for table in tables:
# #             if 'summary' in table[0].lower():
# #                 print(f"  Found: {table[0]}")
# #                 try:
# #                     cols = conn.execute(f'DESCRIBE "{table[0]}"').fetchall()
# #                     print(f"    Columns: {[col[0] for col in cols]}")
# #                 except:
# #                     print("    Could not describe this table")
    
# #     # Check if there are any columns with MRR in the name
# #     print("\n=== SEARCHING FOR MRR COLUMNS ===")
# #     for table_name in [t[0] for t in tables]:
# #         try:
# #             cols = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
# #             mrr_cols = [col[0] for col in cols if 'mrr' in col[0].lower()]
# #             if mrr_cols:
# #                 print(f"Table {table_name} has MRR columns: {mrr_cols}")
                
# #                 # Show sample MRR data
# #                 sample_mrr = conn.execute(f'SELECT run_identifier, {", ".join(mrr_cols)} FROM "{table_name}" LIMIT 5').fetchall()
# #                 for row in sample_mrr:
# #                     print(f"  {row}")
# #         except:
# #             continue
    
# #     print("\n=== LOOKING FOR RANKING STATISTICS TABLES ===")
# #     stats_tables = [t[0] for t in tables if 'summary' in t[0].lower() or 'stats' in t[0].lower()]
# #     for table in stats_tables:
# #         print(f"\nTable: {table}")
# #         try:
# #             cols = conn.execute(f'DESCRIBE "{table}"').fetchall()
# #             col_names = [col[0] for col in cols]
# #             print(f"  Columns: {col_names}")
            
# #             # Show what metrics are available
# #             if 'run_identifier' in col_names:
# #                 models = conn.execute(f'SELECT DISTINCT run_identifier FROM "{table}"').fetchall()
# #                 print(f"  Models: {[m[0] for m in models]}")
# #         except Exception as e:
# #             print(f"  Error: {e}")

# # except Exception as e:
# #     print(f"Database error: {e}")
# # finally:
# #     conn.close()

# # print("\n" + "=" * 60)
# # print("This will show us what MRR data is already available in your database!")
# # print("If MRR columns exist, we should extract them instead of calculating.")

# import duckdb
# from pathlib import Path
# import pandas as pd

# # base directory for the DuckDB database
# BASE_DIR = Path(__file__).parent
# DUCKDB_PATH = BASE_DIR / "LLMs_RAG_VS_NON_RAG_BENCHMARK/LLMsRAG_VS_Non_Benchmark.duckdb"
# conn = duckdb.connect(str(DUCKDB_PATH))

# try:
#     print("LLM PERFORMANCE ANALYSIS - EXTRACTING FROM DATABASE")
#     print("=" * 80)
    
#     # Extract all data from the summary table
#     summary_data = conn.execute('SELECT * FROM "LLMsRAG_VS_Non_RAG_Benchmark_disease_summary" ORDER BY mrr DESC').fetchall()
    
#     # Get column names
#     columns = conn.execute('DESCRIBE "LLMsRAG_VS_Non_RAG_Benchmark_disease_summary"').fetchall()
#     col_names = [col[0] for col in columns]
    
#     # Convert to DataFrame for easier handling
#     df = pd.DataFrame(summary_data, columns=col_names)
    
#     print("=" * 80)
#     print("OVERALL MODEL PERFORMANCE RANKING (by MRR)")
#     print("=" * 80)
    
#     # Main performance table
#     print(f"{'Rank':<4} {'Model':<20} {'MRR':<8} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<8} {'Found':<8}")
#     print("-" * 80)
    
#     for i, (_, row) in enumerate(df.iterrows(), 1):
#         print(f"{i:<4} {row['run_identifier']:<20} {row['mrr']:.3f}    "
#               f"{row['percentage@1']:.1f}%    {row['percentage@3']:.1f}%    "
#               f"{row['percentage@5']:.1f}%    {row['percentage@10']:.1f}%    "
#               f"{row['percentage_found']:.1f}%")
    
#     print("\n" + "=" * 80)
#     print("DETAILED METRICS FOR EACH MODEL")
#     print("=" * 80)
    
#     for _, row in df.iterrows():
#         print(f"\n🤖 {row['run_identifier']}:")
#         print("-" * 50)
        
#         # Core ranking metrics
#         print(f"📊 Ranking Performance:")
#         print(f"   MRR (Mean Reciprocal Rank): {row['mrr']:.3f}")
#         print(f"   Top-1 Accuracy: {row['percentage@1']:.1f}% ({row['top1']}/{row['total']})")
#         print(f"   Top-3 Accuracy: {row['percentage@3']:.1f}% ({row['top3']}/{row['total']})")
#         print(f"   Top-5 Accuracy: {row['percentage@5']:.1f}% ({row['top5']}/{row['total']})")
#         print(f"   Top-10 Accuracy: {row['percentage@10']:.1f}% ({row['top10']}/{row['total']})")
#         print(f"   Found any rank: {row['percentage_found']:.1f}% ({row['found']}/{row['total']})")
        
#         # MAP metrics
#         print(f"📈 Mean Average Precision:")
#         print(f"   MAP@1: {row['MAP@1']:.3f}")
#         print(f"   MAP@3: {row['MAP@3']:.3f}")
#         print(f"   MAP@5: {row['MAP@5']:.3f}")
#         print(f"   MAP@10: {row['MAP@10']:.3f}")
        
#         # NDCG metrics
#         print(f"🎯 Normalized Discounted Cumulative Gain:")
#         print(f"   NDCG@3: {row['NDCG@3']:.3f}")
#         print(f"   NDCG@5: {row['NDCG@5']:.3f}")
#         print(f"   NDCG@10: {row['NDCG@10']:.3f}")
        
#         # Classification metrics
#         print(f"⚖️ Classification Metrics:")
#         print(f"   Accuracy: {row['accuracy']:.3f}")
#         print(f"   F1-Score: {row['f1_score']:.3f}")
#         print(f"   Sensitivity (Recall): {row['sensitivity']:.3f}")
#         print(f"   Specificity: {row['specificity']:.3f}")
#         print(f"   Precision: {row['precision']:.3f}")
#         print(f"   MCC: {row['matthews_correlation_coefficient']:.3f}")
        
#         # Confusion matrix
#         print(f"🔢 Classification Counts:")
#         print(f"   True Positives: {row['true_positives']}")
#         print(f"   False Positives: {row['false_positives']}")
#         print(f"   True Negatives: {row['true_negatives']}")
#         print(f"   False Negatives: {row['false_negatives']}")
    
#     # Summary comparison table
#     print("\n" + "=" * 80)
#     print("SIDE-BY-SIDE COMPARISON TABLE")
#     print("=" * 80)
    
#     comparison_df = df[['run_identifier', 'mrr', 'percentage@1', 'percentage@3', 
#                        'percentage@5', 'percentage@10', 'MAP@10', 'NDCG@10', 
#                        'accuracy', 'f1_score']].round(3)
    
#     print(comparison_df.to_string(index=False))
    
#     # Best performer analysis
#     print("\n" + "=" * 80)
#     print("BEST PERFORMERS BY METRIC")
#     print("=" * 80)
    
#     metrics_to_check = {
#         'MRR': 'mrr',
#         'Top-1 Accuracy': 'percentage@1', 
#         'Top-10 Accuracy': 'percentage@10',
#         'MAP@10': 'MAP@10',
#         'NDCG@10': 'NDCG@10',
#         'Overall Accuracy': 'accuracy',
#         'F1-Score': 'f1_score'
#     }
    
#     for metric_name, column in metrics_to_check.items():
#         best_model = df.loc[df[column].idxmax(), 'run_identifier']
#         best_score = df[column].max()
#         print(f"🏆 Best {metric_name}: {best_model} ({best_score:.3f})")
    
#     # Export to CSV
#     csv_filename = "llm_RAG_performance_summary.csv"
#     df.to_csv(csv_filename, index=False)
#     print(f"\n💾 Full results exported to: {csv_filename}")
    
#     # Key insights
#     print("\n" + "=" * 80)
#     print("KEY INSIGHTS")
#     print("=" * 80)
    
#     best_mrr = df.loc[df['mrr'].idxmax()]
#     worst_mrr = df.loc[df['mrr'].idxmin()]
    
#     print(f"🥇 Best overall performer: {best_mrr['run_identifier']} (MRR: {best_mrr['mrr']:.3f})")
#     print(f"📉 Lowest performer: {worst_mrr['run_identifier']} (MRR: {worst_mrr['mrr']:.3f})")
#     print(f"📊 Performance gap: {(best_mrr['mrr'] - worst_mrr['mrr']):.3f} MRR points")
#     print(f"🎯 Average MRR across all models: {df['mrr'].mean():.3f}")
#     print(f"📈 MRR standard deviation: {df['mrr'].std():.3f}")
    
#     # Show range of top-1 performance
#     print(f"🔍 Top-1 accuracy range: {df['percentage@1'].min():.1f}% - {df['percentage@1'].max():.1f}%")
#     print(f"🔍 Top-10 accuracy range: {df['percentage@10'].min():.1f}% - {df['percentage@10'].max():.1f}%")

# except Exception as e:
#     print(f"Error: {e}")
#     import traceback
#     traceback.print_exc()
# finally:
#     conn.close()

import duckdb
from pathlib import Path
import pandas as pd
import sys

def find_summary_table(conn):
    """Find the summary table in the database"""
    try:
        # Get all table names
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"Found {len(table_names)} tables in database:")
        for i, table in enumerate(table_names, 1):
            print(f"  {i}. {table}")
        
        # Look for summary tables
        summary_tables = [table for table in table_names if 'summary' in table.lower()]
        
        if len(summary_tables) == 1:
            print(f"\nAuto-detected summary table: {summary_tables[0]}")
            return summary_tables[0]
        elif len(summary_tables) > 1:
            print(f"\nMultiple summary tables found:")
            for i, table in enumerate(summary_tables, 1):
                print(f"  {i}. {table}")
            
            while True:
                try:
                    choice = int(input(f"Select summary table (1-{len(summary_tables)}): ")) - 1
                    if 0 <= choice < len(summary_tables):
                        return summary_tables[choice]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            print("\nNo summary table found. Available tables:")
            for i, table in enumerate(table_names, 1):
                print(f"  {i}. {table}")
            
            while True:
                try:
                    choice = int(input(f"Select table to analyze (1-{len(table_names)}): ")) - 1
                    if 0 <= choice < len(table_names):
                        return table_names[choice]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
                    
    except Exception as e:
        print(f"Error finding tables: {e}")
        return None

# Get database path from command line or user input
if len(sys.argv) > 1:
    DUCKDB_PATH = sys.argv[1]
else:
    DUCKDB_PATH = input("Enter path to DuckDB file: ").strip()

# Validate file exists
if not Path(DUCKDB_PATH).exists():
    print(f"Error: File '{DUCKDB_PATH}' not found.")
    sys.exit(1)

conn = duckdb.connect(str(DUCKDB_PATH))

# Find the summary table
table_name = find_summary_table(conn)
if not table_name:
    print("No table selected. Exiting.")
    conn.close()
    sys.exit(1)

try:
    print("\nLLM PERFORMANCE ANALYSIS - EXTRACTING FROM DATABASE")
    print("=" * 80)
    print(f"Database: {DUCKDB_PATH}")
    print(f"Table: {table_name}")
    print("=" * 80)
    
    # Extract all data from the summary table
    summary_data = conn.execute(f'SELECT * FROM "{table_name}" ORDER BY mrr DESC').fetchall()
    
    # Get column names
    columns = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
    col_names = [col[0] for col in columns]
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(summary_data, columns=col_names)
    
    print("=" * 80)
    print("OVERALL MODEL PERFORMANCE RANKING (by MRR)")
    print("=" * 80)
    
    # Main performance table
    print(f"{'Rank':<4} {'Model':<20} {'MRR':<8} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Top-10':<8} {'Found':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i:<4} {row['run_identifier']:<20} {row['mrr']:.3f}    "
              f"{row['percentage@1']:.1f}%    {row['percentage@3']:.1f}%    "
              f"{row['percentage@5']:.1f}%    {row['percentage@10']:.1f}%    "
              f"{row['percentage_found']:.1f}%")
    
    print("\n" + "=" * 80)
    print("DETAILED METRICS FOR EACH MODEL")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n🤖 {row['run_identifier']}:")
        print("-" * 50)
        
        # Core ranking metrics
        print(f"📊 Ranking Performance:")
        print(f"   MRR (Mean Reciprocal Rank): {row['mrr']:.3f}")
        print(f"   Top-1 Accuracy: {row['percentage@1']:.1f}% ({row['top1']}/{row['total']})")
        print(f"   Top-3 Accuracy: {row['percentage@3']:.1f}% ({row['top3']}/{row['total']})")
        print(f"   Top-5 Accuracy: {row['percentage@5']:.1f}% ({row['top5']}/{row['total']})")
        print(f"   Top-10 Accuracy: {row['percentage@10']:.1f}% ({row['top10']}/{row['total']})")
        print(f"   Found any rank: {row['percentage_found']:.1f}% ({row['found']}/{row['total']})")
        
        # MAP metrics
        print(f"📈 Mean Average Precision:")
        print(f"   MAP@1: {row['MAP@1']:.3f}")
        print(f"   MAP@3: {row['MAP@3']:.3f}")
        print(f"   MAP@5: {row['MAP@5']:.3f}")
        print(f"   MAP@10: {row['MAP@10']:.3f}")
        
        # NDCG metrics
        print(f"🎯 Normalized Discounted Cumulative Gain:")
        print(f"   NDCG@3: {row['NDCG@3']:.3f}")
        print(f"   NDCG@5: {row['NDCG@5']:.3f}")
        print(f"   NDCG@10: {row['NDCG@10']:.3f}")
        
        # Classification metrics
        print(f"⚖️ Classification Metrics:")
        print(f"   Accuracy: {row['accuracy']:.3f}")
        print(f"   F1-Score: {row['f1_score']:.3f}")
        print(f"   Sensitivity (Recall): {row['sensitivity']:.3f}")
        print(f"   Specificity: {row['specificity']:.3f}")
        print(f"   Precision: {row['precision']:.3f}")
        print(f"   MCC: {row['matthews_correlation_coefficient']:.3f}")
        
        # Confusion matrix
        print(f"🔢 Classification Counts:")
        print(f"   True Positives: {row['true_positives']}")
        print(f"   False Positives: {row['false_positives']}")
        print(f"   True Negatives: {row['true_negatives']}")
        print(f"   False Negatives: {row['false_negatives']}")
    
    # Summary comparison table
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON TABLE")
    print("=" * 80)
    
    comparison_df = df[['run_identifier', 'mrr', 'percentage@1', 'percentage@3', 
                       'percentage@5', 'percentage@10', 'MAP@10', 'NDCG@10', 
                       'accuracy', 'f1_score']].round(3)
    
    print(comparison_df.to_string(index=False))
    
    # Best performer analysis
    print("\n" + "=" * 80)
    print("BEST PERFORMERS BY METRIC")
    print("=" * 80)
    
    metrics_to_check = {
        'MRR': 'mrr',
        'Top-1 Accuracy': 'percentage@1', 
        'Top-10 Accuracy': 'percentage@10',
        'MAP@10': 'MAP@10',
        'NDCG@10': 'NDCG@10',
        'Overall Accuracy': 'accuracy',
        'F1-Score': 'f1_score'
    }
    
    for metric_name, column in metrics_to_check.items():
        best_model = df.loc[df[column].idxmax(), 'run_identifier']
        best_score = df[column].max()
        print(f"🏆 Best {metric_name}: {best_model} ({best_score:.3f})")
    
    # Export to CSV
    csv_filename = f"llm_performance_summary_{Path(DUCKDB_PATH).stem}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Full results exported to: {csv_filename}")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    best_mrr = df.loc[df['mrr'].idxmax()]
    worst_mrr = df.loc[df['mrr'].idxmin()]
    
    print(f"🥇 Best overall performer: {best_mrr['run_identifier']} (MRR: {best_mrr['mrr']:.3f})")
    print(f"📉 Lowest performer: {worst_mrr['run_identifier']} (MRR: {worst_mrr['mrr']:.3f})")
    print(f"📊 Performance gap: {(best_mrr['mrr'] - worst_mrr['mrr']):.3f} MRR points")
    print(f"🎯 Average MRR across all models: {df['mrr'].mean():.3f}")
    print(f"📈 MRR standard deviation: {df['mrr'].std():.3f}")
    
    # Show range of top-1 performance
    print(f"🔍 Top-1 accuracy range: {df['percentage@1'].min():.1f}% - {df['percentage@1'].max():.1f}%")
    print(f"🔍 Top-10 accuracy range: {df['percentage@10'].min():.1f}% - {df['percentage@10'].max():.1f}%")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()