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

cursor = decl.walk()
if cursor.goto_first_child():
    while True:
        if cursor.field_name == 'declarator':
            find_id(cursor.node)
        if not cursor.goto_next_sibling():
            break

print("Local vars found:", local_vars)
