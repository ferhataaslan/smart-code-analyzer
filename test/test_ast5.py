import tree_sitter_cpp
from tree_sitter import Language, Parser
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser(CPP_LANGUAGE)
code = b"const char *host = NULL, *port = NULL, *path = NULL;"
tree = parser.parse(code)
decl = tree.root_node.children[0].children[0] # declaration

local_vars = set()
source_bytes = code

def find_id(node):
    if node.type == 'identifier':
        local_vars.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
    else:
        for c in node.children:
            if node.type == 'init_declarator' and c == node.child_by_field_name('value'):
                continue
            find_id(c)

for child in decl.children:
    if child.type in ['init_declarator', 'pointer_declarator', 'array_declarator', 'function_declarator', 'reference_declarator', 'identifier']:
        find_id(child)

print("Local vars found:", local_vars)
