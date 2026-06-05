import tree_sitter, tree_sitter_cpp
parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language()))
code = b'''
int my_func(int arg1, char* arg2, int& arg3) {
    int local_var = 5;
    int *ptr = &local_var;
    int &ref = local_var;
    int arr[10];
    printf("hello");
    return local_var + arg1;
}
'''
tree = parser.parse(code)
def get_local_identifiers(root, source_bytes):
    local_vars = set()
    func_names = set()
    
    def walk(n):
        if n.type == 'function_declarator':
            decl = n.child_by_field_name('declarator')
            if decl and decl.type == 'identifier':
                func_names.add(source_bytes[decl.start_byte:decl.end_byte].decode('utf-8'))
        
        if n.type in ['parameter_declaration', 'declaration']:
            decl = n.child_by_field_name('declarator')
            if decl:
                def find_id(node):
                    if node.type == 'identifier':
                        local_vars.add(source_bytes[node.start_byte:node.end_byte].decode('utf-8'))
                    else:
                        for c in node.children:
                            if node.type == 'init_declarator' and c == node.child_by_field_name('value'):
                                continue
                            find_id(c)
                find_id(decl)
                
        for c in n.children:
            walk(c)
            
    walk(root)
    return local_vars, func_names

print(get_local_identifiers(tree.root_node, code))
