import sqlite3
import pandas as pd

def check_db():
    print("--- VECTOR STORE SCHEMA ---")
    try:
        conn = sqlite3.connect("data/dbs/vector_store.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(vectors);")
        for row in cursor.fetchall():
            print(row)
        
        print("\n--- SAMPLE ROW (VECTOR STORE) ---")
        cursor.execute("SELECT entity_id, length(normalized_structure) FROM vectors LIMIT 1;")
        print(cursor.fetchone())
        conn.close()
    except Exception as e:
        print(e)

    print("\n--- DOCUMENT STORE SCHEMA ---")
    try:
        conn = sqlite3.connect("data/dbs/document_store.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(documents);")
        for row in cursor.fetchall():
            print(row)
            
        print("\n--- SAMPLE ROW (DOCUMENT STORE) ---")
        cursor.execute("SELECT entity_id, length(raw_snippet) FROM documents LIMIT 1;")
        print(cursor.fetchone())
        conn.close()
    except Exception as e:
        print(e)
        
    print("\n--- PARQUET COLUMNS ---")
    try:
        import glob
        parquet_files = glob.glob("*.parquet")
        if parquet_files:
            df = pd.read_parquet(parquet_files[-1])
            print(f"Columns in {parquet_files[-1]}:")
            print(df.dtypes)
            print("\nSample values length:")
            print("raw_snippet length:", len(str(df['raw_snippet'].iloc[0])) if 'raw_snippet' in df else "N/A")
            print("normalized_structure length:", len(str(df['normalized_structure'].iloc[0])) if 'normalized_structure' in df else "N/A")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    check_db()
