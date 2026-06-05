import tree_sitter, tree_sitter_cpp
parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language()))
code = b'''
int my_func(int arg1, char* arg2) {
    int local_var = 5;
    printf("hello");
    return local_var + arg1;
}
'''
tree = parser.parse(code)
query = tree_sitter.Query(tree_sitter.Language(tree_sitter_cpp.language()), '''
(function_declarator declarator: (identifier) @func_name)
(parameter_declaration declarator: (identifier) @var_decl)
(parameter_declaration declarator: (pointer_declarator declarator: (identifier) @var_decl))
(declaration declarator: (identifier) @var_decl)
(declaration declarator: (init_declarator declarator: (identifier) @var_decl))
(declaration declarator: (init_declarator declarator: (pointer_declarator declarator: (identifier) @var_decl)))
(declaration declarator: (array_declarator declarator: (identifier) @var_decl))
''')
cursor = tree_sitter.QueryCursor(query)
for _, c in cursor.matches(tree.root_node):
    for tag, nodes in c.items():
        if not isinstance(nodes, list): nodes = [nodes]
        for n in nodes:
            print(tag, code[n.start_byte:n.end_byte].decode())
