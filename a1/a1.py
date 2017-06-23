def run_fsa(states, initial_state, accept_states, transition, input_symbols):
    """
    Implement a deterministic finite-state automata.
    See test_baa_fsa.
    Params:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
      input_symbols...list of strings representing the input string
    Returns:
      True if this FSA accepts the input string; False otherwise.
    """
    ###TODO
    
        
    state = initial_state
    total = len(input_symbols)
    #print(total)
    #print(input_symbols)
    while total > 0:
        
        f_element= input_symbols.pop(0)
        if state in accept_states:
            return False
        
        if state in transition.keys() and f_element in transition[state]:
            state = transition[state][f_element]
        else:
            return False
        
        
        total -=1
    
        
        
    if state in accept_states:
        return True
    return False
    ###

def get_name_fsa():
    """
    Define a deterministic finite-state machine to recognize a small set of person names, such as:
      Mr. Frank Michael Lewis
      Ms. Flo Lutz
      Frank Micael Lewis
      Flo Lutz
    See test_name_fsa for examples.
    Names have the following:
    - an optional prefix in the set {'Mr.', 'Ms.'}
    - a required first name in the set {'Frank', 'Flo'}
    - an optional middle name in the set {'Michael', 'Maggie'}
    - a required last name in the set {'Lewis', 'Lutz'}
    Returns:
      A 4-tuple of variables, in this order:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
    """
    ###TODO
    states = [0, 1, 2, 3, 4]
    initial_state = 0
    accept_states = [4]
    li = [['Mr.', 'Ms.'],['Frank', 'Flo'],['Michael', 'Maggie'],['Lewis', 'Lutz']]
    transition = {
        0: {li[0][0]: 1, li[0][1]: 1, li[1][0]: 2, li[1][1]: 2},
        1: {li[1][0]: 2, li[1][1]: 2},
        2: {li[2][0]: 3, li[2][1]: 3, li[3][0]: 4, li[3][1]: 4},
        3: {li[3][0]: 4, li[3][1]: 4}        
    }
    return states,initial_state, accept_states, transition


def read_grammar(lines):
    """Read a list of strings representing CFG rules. E.g., the string
    'S :- NP VP'
    should be parsed into a tuple
    ('S', ['NP', 'VP'])
    Note that the first element of the tuple is a string ('S') for the
    left-hand-side of the rule, and the second element is a list of
    strings (['NP', 'VP']) for the right-hand-side of the rule.
    
    See test_read_grammar.
    Params:
      lines...A list of strings, one per rule
    Returns:
      A list of (LHS, RHS) tuples, one per rule.
    """
    ###TODO
    
    li = []
    for i in lines:
        l_x = i.split(':-')
        l_x[1] = l_x[1].split()
        #print(l_x[1])
        li.append((l_x[0].strip(), l_x[1]))
        #print(li)
    return li
def is_pos(rule,rules):
    
    
    """
    Returns:
      True if this rule is a part-of-speech rule, which is true if none of the
      RHS symbols appear as LHS symbols in other rules.
    E.g., if the grammar is:
    S :- NP VP
    NP :- N
    VP :- V
    N :- dog cat
    V :- run likes
    Then the final two rules are POS rules.
    See test_is_pos.
    This function should be used by the is_valid_production function
    below.
    
    """
    """
    rules = [('S', ['NP', 'VP']),
            ('NP', ['ProperNoun']),
            ('ProperNoun', ['John', 'Mary']),
            ('VP', ['V', 'ProperNoun']),
            ('V', ['likes', 'hates'])]
    """
    
    pos = []
    for i in rules:
        pos.append(i[0])
    for j in rule[1]:
        if j in pos:
            return False
    return True
    
def is_valid_production(production, rules):
    """
    Params:
      production...A (LHS, RHS) tuple representing one production,
                   where LHS is a string and RHS is a list of strings.
      rules........A list of tuples representing the rules of the grammar.
    Returns:
      True if this production is valid according to the rules of the
      grammar; False otherwise.
    See test_is_valid_production.
    This function should be used in the is_valid_tree method below.
    """
    '''
    li = []
    for i in rules:
        li.append(i[0])
    #print(li)
    li1 = []   
    for j in rules:
        li1.append(j[1])
        #print(li1)
    my_dict_12 = dict(zip(li,li1))
    #print(my_dict_12)
    #for i in rules:
    #print(i)
    #print(my_dict_12[production[0]][1])
    #print(production[0][0])
    li_b = my_dict_12[production[0]]
    #print(li_b)
    if (my_dict_12[production[0]] == production[1]):
        return True
    if (my_dict_12[production[0]] == production[1][0]):
        
        return True
    #print(len(my_dict_12[production[0]])
    if len(production[1]) < 2:
        for i in li_b:
            if i in production[1]:
                return True
    return False
    '''
    li =[]
    for i in rules:
        
        if i[0] == production[0]:
            li.append(i)
    for j in li:
        if production[1] == j[1]:
            return True
        if is_pos(j,rules):
            if production[1][0] in j[1]:
                return True
        
    return False
        

def is_valid_tree(tree, rules, words):
    """
    Params:
      tree....A Tree object representing a parse tree.
      rules...The list of rules in the grammar.
      words...A list of strings representing the sentence to be parsed.

    Returns:
      True if the tree is a valid parse of this sentence. This requires:
        - every production in the tree is valid (present in the list of rules).
        - the leaf nodes in the tree match the words in the sentence, in order.
    
    See test_is_valid_tree.
    """
    ###TODO
    a = tree.get_leaves()
    #print(a)
    if a !=words:
        return False
    
    
    b = tree.get_productions()
    #print(b)
    for i in b:
        if not is_valid_production(i,rules):
            return False
    return True
            
        
    
        
    
    
    
    
class Tree:
    """A partial implementation of a Tree class to represent a parse tree.
    Each node in the Tree is also a Tree.
    Each Tree has two attributes:
      - label......a string representing the node (e.g., 'S', 'NP', 'dog')
      - children...a (possibly empty) list of children of this
                   node. Each element of this list is another Tree.

    A leaf node is a Tree with an empty list of children.
    """
    def __init__(self, label, children=[]):
        """The constructor.
        Params:
          label......A string representing this node
          children...An optional list of Tree nodes, representing the
                     children of this node.
        This is done for you and should not be modified.
        """
        self.label = label
        self.children = children

    def __str__(self):
        """
        Print a string representation of the tree, for debugging.
        This is done for you and should not be modified.
        """
        s = self.label
        for c in self.children:
            s += ' ( ' + str(c) + ' ) '
        return s

    def get_leaves(self):
        """
        Returns:
          A list of strings representing the leaves of this tree.
        See test_get_leaves.
        """
        ###TODO
        li = []
        
        
        #li1 = []
        for i in self.children:
            li.extend(i.get_leaves())
        #print(li)
        if len(self.children) == 0:
            li.append(self.label)
        return li
        
        

    def get_productions(self):
        """Returns:
          A list of tuples representing a depth-first traversal of
          this tree.  Each tuple is of the form (LHS, RHS), where LHS
          is a string representing the left-hand-side of the
          production, and RHS is a list of strings representing the
          right-hand-side of the production.

        See test_get_productions.
        """
        ###TODO
        
        li =[]
        if len(self.children) !=0:    
            li1 = []
            for i in self.children:
                li1.append(i.label)
            li.append((self.label,li1))
        
            
        for i in self.children:
            li.extend(i.get_productions())
            
        
        
        return li
        
        
        

        
    
