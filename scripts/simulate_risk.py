import json
from datasets import load_dataset
import statistics

print('Dataset yükleniyor (streaming)...')
ds = load_dataset('smart-code-analyzer-team/cpp-vulnerability-dataset', split='train', streaming=True)

dfg_edge_counts = []
ast_branch_counts = []

count = 0
for rec in ds:
    if count >= 1000:
        break
        
    # Analyze DFG
    raw_dfg = rec.get('data_flow_graph', '')
    dfg_edges = 0
    if raw_dfg and raw_dfg.strip() not in ('', 'null', 'None'):
        try:
            dfg = json.loads(raw_dfg)
            if isinstance(dfg, list):
                # If DFG is a list of edges/tuples
                dfg_edges = len(dfg)
            elif isinstance(dfg, dict):
                # Maybe dict with 'edges' key
                dfg_edges = len(dfg.get('edges', []))
        except:
            pass
    dfg_edge_counts.append(dfg_edges)
    
    # Analyze AST
    raw_ast = rec.get('ast_metadata', '')
    ast_branches = 0
    if raw_ast and raw_ast.strip() not in ('', 'null', 'None'):
        try:
            ast = json.loads(raw_ast)
            ast_str = json.dumps(ast)
            ast_branches = ast_str.count('"if_statement"') + ast_str.count('"for_statement"') + ast_str.count('"while_statement"') + ast_str.count('"switch_statement"')
        except:
            pass
    ast_branch_counts.append(ast_branches)
    
    count += 1

print(f'\n--- DFG Edges (Data Flow Yoğunluğu) ---')
print(f'Max: {max(dfg_edge_counts)}')
print(f'Mean: {statistics.mean(dfg_edge_counts):.2f}')
print(f'Median: {statistics.median(dfg_edge_counts)}')
print(f'Zero Count: {dfg_edge_counts.count(0)} / {count}')

print(f'\n--- AST Branches (Mantıksal Dallanma Yoğunluğu) ---')
print(f'Max: {max(ast_branch_counts)}')
print(f'Mean: {statistics.mean(ast_branch_counts):.2f}')
print(f'Median: {statistics.median(ast_branch_counts)}')
print(f'Zero Count: {ast_branch_counts.count(0)} / {count}')
