#!/usr/bin/env python

import os
import sys
import pandas as pd

def detect_conflicts(df1, df2, index_name="BASIC_INFO_ID"):
    df1 = df1.set_index(index_name) if df1.index.name != index_name else df1
    df2 = df2.set_index(index_name) if df2.index.name != index_name else df2

    common_index = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)

    df1_aligned = df1.reindex(index=common_index, columns=common_columns).round(5)
    df2_aligned = df2.reindex(index=common_index, columns=common_columns).round(5)

    conflict_columns = []
    for col in common_columns:
        s1 = df1_aligned.loc[common_index, col]
        s2 = df2_aligned.loc[common_index, col]
        mask = s1.notna() & s2.notna()

        if not s1[mask].eq(s2[mask]).all(): 
            if not col.startswith("RESTING_"):
                print(col)
                for idx in common_index[mask]:
                    if s1.loc[idx] != s2.loc[idx]:
                        print(idx)
                        print(s1.loc[idx])
                        print(s2.loc[idx])
                        print()
                        break
            conflict_columns.append(col)

    return conflict_columns

def union_dataframe(df1, df2, index_name="BASIC_INFO_ID"):
    df1 = df1.set_index(index_name) if df1.index.name != index_name else df1
    df2 = df2.set_index(index_name) if df2.index.name != index_name else df2

    union_index = df1.index.union(df2.index)
    union_columns = df1.columns.union(df2.columns)

    df1_aligned = df1.reindex(index=union_index, columns=union_columns)
    df2_aligned = df2.reindex(index=union_index, columns=union_columns)

    df_merged = df2_aligned.combine_first(df1_aligned)
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={"index": index_name}, inplace=True)

    return df_merged

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide the date of the new file as a command line argument (YYYY-MM-DD)."
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rawdata")

    new_file_date = sys.argv[1]
    out_path = os.path.join(data_dir, f"DATA_ses-01_{new_file_date}.csv")
    
    df1 = pd.read_csv(os.path.join(data_dir, "DATA_ses-01_2024-12-09.csv"))
    df2 = pd.read_csv(os.path.join(data_dir, "DATA_ses-01_2025-05-29.csv"))
    
    conflicts = detect_conflicts(df1, df2)
    if conflicts:
        print(f"{len(conflicts)} conflicting columns detected.")

    combined_df = union_dataframe(df1, df2)
    combined_df.to_csv(out_path, index=False)
    print(f"\nCombined DataFrame saved to:\n{out_path}\n")
