#!/usr/bin/env python3
import pandas as pd
import json
import glob
import os

def inspect_dataset():
    print("=" * 80)
    print("  DATASET INSPECTION UTILITY")
    print("=" * 80)

    # Find parquet files in the current directory
    parquet_files = glob.glob("data/parquets/data_Casper_*.parquet")
    if not parquet_files:
        print("No Casper parquet files found in current directory.")
        return

    # Sort files to find the latest one
    parquet_files.sort()
    latest_file = parquet_files[-1]
    print(f"Loading latest dataset file: {latest_file}")

    try:
        df = pd.read_parquet(latest_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    print(f"Dataset shape: {df.shape} (rows, columns)")
    print(f"Columns: {list(df.columns)}")
    print("-" * 80)

    # Count by source system
    if "source_system" in df.columns:
        print("Record counts by source_system:")
        print(df["source_system"].value_counts())
        print("-" * 80)

    # Count by complexity
    if "complexity" in df.columns:
        print("Record counts by complexity:")
        print(df["complexity"].value_counts())
        print("-" * 80)

    # Inspect column types and check if there are null values
    print("Column Types:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    print("-" * 80)

    # Preview different source systems
    for source in df["source_system"].unique():
        sub_df = df[df["source_system"] == source]
        if len(sub_df) == 0:
            continue
        print(f"Previewing record from source: {source}")
        sample = sub_df.iloc[0]
        
        print(f"  Entity ID: {sample.get('entity_id', 'N/A')}")
        print(f"  Applied Algorithm: {sample.get('applied_algorithm', 'N/A')}")
        print(f"  Complexity: {sample.get('complexity', 'N/A')}")
        
        # Raw snippet preview
        raw_code = sample.get('raw_snippet', '')
        print(f"  Raw Snippet (first 150 chars):\n{raw_code[:150]}...")
        
        # Normalized structure preview
        norm_struct = sample.get('normalized_structure', '')
        print(f"  Normalized Structure (first 250 chars):\n{norm_struct[:250]}...")
        
        # Check if normalized_structure is JSON
        is_json = False
        try:
            parsed = json.loads(norm_struct)
            is_json = True
            print("  [!] normalized_structure is a JSON object! Keys:", list(parsed.keys()))
        except Exception:
            pass

        if not is_json:
            print("  [OK] normalized_structure is raw text/code.")

        # Security context preview
        sec_ctx = sample.get('security_context')
        if sec_ctx is not None:
            # Check if it's dict or string
            if isinstance(sec_ctx, str):
                print(f"  Security Context (string, first 200 chars):\n{sec_ctx[:200]}...")
            elif isinstance(sec_ctx, dict):
                print(f"  Security Context (Dict): Keys: {list(sec_ctx.keys())}")
                if "risk_scoring" in sec_ctx:
                    print("    Risk scoring:", sec_ctx["risk_scoring"])

        # Data flow graph preview
        dfg = sample.get('data_flow_graph')
        if dfg is not None:
            if isinstance(dfg, str):
                print(f"  Data Flow Graph (string, first 200 chars):\n{dfg[:200]}...")
            else:
                print(f"  Data Flow Graph (List of items): count = {len(dfg)}")

        print("=" * 80)

if __name__ == "__main__":
    inspect_dataset()
