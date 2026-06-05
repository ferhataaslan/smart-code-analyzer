import tree_sitter_cpp
from tree_sitter import Language, Parser
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser(CPP_LANGUAGE)
code = b"BlockDriverState *bs;"
tree = parser.parse(code)
def print_tree(node, depth=0):
    print("  " * depth + f"{node.type} (field: {node.parent.field_name_for_child(node.parent.children.index(node)) if node.parent else None})")
    for child in node.children:
        print_tree(child, depth + 1)
print_tree(tree.root_node)
