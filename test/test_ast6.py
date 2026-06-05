import tree_sitter_cpp
from tree_sitter import Language, Parser
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser(CPP_LANGUAGE)
code = b"const char *host = NULL, *port = NULL, *path = NULL;"
tree = parser.parse(code)
decl = tree.root_node.children[0].children[0]
for child in decl.children:
    print(child.type)
