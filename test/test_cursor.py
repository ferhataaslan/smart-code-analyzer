import tree_sitter_cpp
from tree_sitter import Language, Parser
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser(CPP_LANGUAGE)
code = b"const char *host = NULL, *port = NULL, *path = NULL;"
tree = parser.parse(code)
decl = tree.root_node.children[0].children[0] # declaration
cursor = decl.walk()
if cursor.goto_first_child():
    while True:
        print(cursor.node.type, cursor.current_field_name())
        if not cursor.goto_next_sibling():
            break
