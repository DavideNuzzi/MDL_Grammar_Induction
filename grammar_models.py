import math
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib as mpl


class GrammarModel:

    def __init__(self, sequence, separator, mdl_model_cost=1, rule_len_min=2, rule_len_max=2):

        self.sequence = sequence
        self.separator = separator
        self.rule_len_min = rule_len_min
        self.rule_len_max = rule_len_max
        self.mdl_model_cost = mdl_model_cost

        # The vocabulary is just the set of unique symbols
        self.vocabulary = set(sequence) 

        # The grammar will store rules of type { 'RULE_1': ('A', 'B'), 'RULE_2': ('RULE_1', 'B', 'C'), ... }
        self.grammar = {} 

        # To name the new rules
        self.rule_counter = 0

        # For debugging purposes, store the number of prunes done in the last fit
        self.rules_pruned_num = 0

        # Store some information about the input data
        self.sequence_init = sequence
        self.vocabulary_init = self.vocabulary
        

    def compute_description_length(self, sequence, grammar, vocabulary):

        # Since we always prune the unused symbols the vocabulary contains all the symbols
        # needed for the sequence and the grammar description. The cost to encode the 
        # vocabulary is computed by a simple bit encoding
        bits_per_symbol_vocab = np.log2(len(vocabulary))

        # The cost of the data is just the average cost per symbol multiplied by the sequence length
        cost_data = len(sequence) * bits_per_symbol_vocab

        # The cost of encoding the grammar is similar, but for each rule we have a different number 
        # of symbols. We whould also add 1 additional symbol to each rule as a "separator", to know 
        # when a rule ends and another one starts. We can ignore the rule name itself, as they are
        # already encoded in the position in the list
        if len(grammar) > 0:
        
            # Count, for each rule, the number of symbols and add 1 for a separator
            total_symbols = sum(len(v) + 1 for v in grammar.values())
            cost_grammar =  total_symbols * bits_per_symbol_vocab
        else:
            cost_grammar = 0

        return cost_data, cost_grammar


    def replace_symbols(self, sequence, rule):

        # The rule is a tuple like (rule_name, rule_n_gram)
        new_symbol, n_gram = rule
        n_gram = list(n_gram)
        n = len(n_gram)

        sequence_new = []

        # Repeat until the end of the sequence
        i = 0
        while i < len(sequence):
            
            # Check if there's enough space
            if i <= len(sequence) - n:
                # Check if there's a match
                if sequence[i:(i+n)] == n_gram:
                    sequence_new.append(new_symbol)
                    i += n
                else:
                    sequence_new.append(sequence[i])
                    i += 1  
            else:
                sequence_new.append(sequence[i])
                i += 1  

        return sequence_new
    

    def add_rule(self, sequence, vocabulary, grammar, rule):

        # Here we compute all the consequences of adding a new rule
        # The rule is a tuple (rule_name, n_gram)

        # First, we copy all the data
        grammar_new = grammar.copy()
        vocabulary_new = vocabulary.copy()
        
        # Then we compute the new sequence
        sequence_new = self.replace_symbols(sequence, rule)

        # And add the element to the vocabulary and grammar
        grammar_new[rule[0]] = rule[1]
        vocabulary_new.add(rule[0])

        return sequence_new, vocabulary_new, grammar_new

    
    def get_possible_rules(self, sequence):

        # Generate all n-grams of length n from the min to max rule length
        # Note that those contain duplicates, i.e. they form a sequence
        candidate_n_grams = []
        for n in range(self.rule_len_max, self.rule_len_min-1, -1):
        # for n in range(self.rule_len_min, self.rule_len_max+1):
            # Efficiently generate n-grams of length n
            candidate_n_grams.extend(zip(*(sequence[i:] for i in range(n))))
            
        # Filter out grams containing the separator
        candidate_n_grams = [g for g in candidate_n_grams if self.separator not in g]

        # Count all occurrences across all lengths
        counts = Counter(candidate_n_grams)

        if not counts:
            return [], np.array([])
        
        # Extract and sort by frequency
        ordered_counts = counts.most_common()

        # Unzip into tuples and counts (the tuples are unique now)
        candidate_n_grams, counts_values = zip(*ordered_counts)

        # # We know that the change in MDL is proportional to both the count and the symbols in the n_gram
        # # So we must correct for this
        # n_grams_lengths = [len(g) for g in candidate_n_grams]
        # counts_corrected = np.array(counts_values) * np.array(n_grams_lengths)

        return list(candidate_n_grams), counts_values
        
    
    def prune_rules(self, sequence, vocabulary, grammar, verbose=False):

        # Figure out which symbols are in the vocabulary but not in the sequence
        unique_symbols_sequence = set(sequence)
        obsolete_symbols = vocabulary.difference(unique_symbols_sequence)

        # Remove from the obsolete symbols the original ones, as we never want to prune them
        obsolete_symbols = obsolete_symbols.difference(self.vocabulary_init)

        # If there are no obsolete symbols, do nothing
        if len(obsolete_symbols) == 0: return vocabulary, grammar

        if verbose:
            print(f'Found that the symbols: {obsolete_symbols} are obsolete and must be pruned')

        # Now we must remove the obsolete symbols from the vocabulary
        vocabulary = vocabulary.difference(obsolete_symbols)

        # And also remove them from the grammar, replacing them with the corresponding rule
        # wherever they appear. Note that we have for sure a definition of the symbol itself, which can
        # be stored and removed at the beginning
        rules_to_prune = {s: grammar[s] for s in obsolete_symbols}
        grammar = {s: grammar[s] for s in grammar if s not in obsolete_symbols}

        if verbose:
            print(f'Replacing the rules: {rules_to_prune} wherever they occur in the grammar')
            
        self.rules_pruned_num += len(rules_to_prune)

        # Note: it may happen that the rules to prune reference themselves, so after a single
        # loop of replacements we did not remove them completely (if rule_2 references rule_1
        # and we first remove all instances of rule_1, then we introduce rule_1 again when we
        # replace all instances of rule_2). This means that we must repeat the cycle for as 
        # long as it takes
        replacements_done = -1
        while replacements_done != 0:
            replacements_done = 0
            
            # Loop over the grammar
            for rule in grammar: 
                rule_values = grammar[rule]

                # Loop over the symbols to prune
                for symbol_to_prune, replacement_rule in rules_to_prune.items():

                    # Check if this symbol is in the rule
                    if symbol_to_prune in rule_values:
                        
                        # Build the new tuple (the new rule, using the symbols of the rule to prune)
                        rule_new = []
                        for r in rule_values:
                            if r == symbol_to_prune: rule_new.extend(replacement_rule)
                            else: rule_new.append(r)

                        if verbose:
                            print(f'Replaced the rule: {rule} -> {grammar[rule]}')

                        grammar[rule] = tuple(rule_new)
                        replacements_done += 1

                        if verbose:
                            print(f'with the rule: {rule} -> {grammar[rule]}')

        return vocabulary, grammar


    def fit(self, rule_label='rule', max_iterations=10000, grammar_max_length=None, verbose=False):

        self.rules_pruned_num = 0
        
        # Compute the minimum description length (MDL) of the current solution
        cost_data, cost_grammar = self.compute_description_length(self.sequence, self.grammar, self.vocabulary)
        mdl = cost_data + self.mdl_model_cost * cost_grammar

        # Repeat for a fixed number of iterations
        pbar = tqdm(range(max_iterations))

        if verbose:
            print(f'Starting MDL = {mdl:.2f} (data = {cost_data:.2f}, grammar = {cost_grammar:.2f})')
            print(f'The sequence is of length {len(self.sequence)}, it contain {len(np.unique(self.sequence))} unique symbols.\nThe vocabulary is of size {len(self.vocabulary)}')
            print('='*80)

        just_found = True
        
        for n in pbar:
            
            # Get all the possible n-grams for the new rule. This should be done only when we actually create a new rule
            # Otherwise we can keep it (as it does not change) when we don't find a good rule
            if just_found:
                n_grams, counts = self.get_possible_rules(self.sequence)
                just_found = False
                
                # If there are no more possible pairs stop
                if len(counts) == 0:
                    break
                
            # Choose the best possible rule
            chosen_n_gram = n_grams[0]
            if verbose: print(f'Chosen the n-gram {chosen_n_gram}, with {counts[0]} occurrences in a sequence of length {len(self.sequence)}')

            # Given the chosen pair, propose a new rule
            new_rule_name = f"{rule_label}_{self.rule_counter+1}"
            new_rule = (new_rule_name, chosen_n_gram)

            # Now test the new rule
            seq_new, vocab_new, grammar_new = self.add_rule(self.sequence, self.vocabulary, self.grammar, new_rule)

            # Compute the new MDL
            cost_data_new, cost_grammar_new = self.compute_description_length(seq_new, grammar_new, vocab_new)
            mdl_new = cost_data_new + self.mdl_model_cost * cost_grammar_new

            if verbose: 
                print(f'The new MDL would be {mdl_new:.2f} (data = {cost_data_new:.2f}, grammar = {cost_grammar_new:.2f})')

            # If it decreases, do the swap with the new rule
            if mdl_new < mdl:
                mdl = mdl_new
                self.sequence = seq_new
                self.vocabulary = vocab_new
                self.grammar = grammar_new
                self.rule_counter += 1

                just_found = True

                if verbose: print(f'Accepted the new rule {new_rule}.\nThe new sequence is of length {len(self.sequence)}, it contain {len(np.unique(self.sequence))} unique symbols.\nThe new vocabulary is of size {len(self.vocabulary)} and the total rules in the grammare are {len(self.grammar)}')
            
                # Prune if needed
                self.vocabulary, self.grammar = self.prune_rules(self.sequence, self.vocabulary, self.grammar, verbose=verbose)
                
                # Check if we must stop
                if grammar_max_length is not None:
                    if len(self.grammar) == grammar_max_length:
                        if verbose: print(f'Grammar has reached the maximum size of {grammar_max_length} so we stop.')
                        break
            else:
                if verbose: print('Since the methods is greedy and the rule change was rejected, we stop.')
                break

            if verbose: print('-'*80)

            if not verbose:
                pbar.set_postfix({'mdl': mdl, 'mdl_proposed': mdl_new, 'Seq len': len(self.sequence), 'vocab len': len(self.vocabulary), 'grammar len': len(self.grammar)})


    def get_grammar_graph(self):

        # The graph is directed (from rule name to substitution)
        graph = nx.DiGraph()
        
        for rule_def, rule_sub in self.grammar.items():
            for r in rule_sub:
                graph.add_edge(rule_def, r)

        # NOTE: we are only relying on the grammar, meaning that
        # if no rule actually includes some terminal symbols they won't be considered
        return graph
    
    # Get the counts of each symbol in the sequence
    # It includes the terminal symbols (even if the count is zero), but ignores separators
    def get_counts(self):
        counts = Counter(self.sequence).most_common()
        symbols = [c[0] for c in counts]

        for sym in self.vocabulary_init:
            if sym not in symbols:
                counts.append((sym, 0))

        counts = {c[0]: c[1] for c in counts if c[0] != self.separator}
        return counts


    # Count all the elements, but not only by their appearence in the sequence
    # but also by their implicit appearence through other symbols + grammar rules
    def get_recursive_counts(self):
        
        # Start with the actual symbols in the sequence
        total_counts = Counter(self.sequence)
        
        # Get the grammar depth to process rules from top to bottom
        depths = self.get_grammar_depth()
        
        # Reverse the order of the depths, as we need to start from the top of the hierarchy
        sorted_rules = sorted(
            [node for node in self.grammar.keys()], 
            key=lambda x: depths[x], 
            reverse=True
        )

        for rule_name in sorted_rules:

            # How many times does this specific rule exist in total (so far)?
            count_of_this_rule = total_counts.get(rule_name, 0)
            
            if count_of_this_rule > 0:
                # Look at what this rule is composed of
                components = self.grammar[rule_name]
                
                # For every symbol inside this rule, add the parent's count to it
                for symbol in components:
                    total_counts[symbol] += count_of_this_rule

        return total_counts


    # Return the depth of each element of the grammar (the distance from the terminal symbols)
    def get_grammar_depth(self, G=None):
        
        # Init the graph to make the computation simpler
        if G is None:
            G = self.get_grammar_graph()

        # Initialize terminals with level 0
        levels = {n: 0 for n in G.nodes() if G.out_degree(n) == 0}
        
        # Iteratively resolve levels for parents
        # (Loop ensures we handle deep dependencies)
        for _ in range(len(G.nodes())):
            changed = False
            for n in G.nodes():
                if n in levels:
                    continue
                
                children = list(G.successors(n))
                # If all children have a known level, we can calculate the parent's level
                if all(c in levels for c in children):
                    levels[n] = 1 + max(levels[c] for c in children)
                    changed = True
            if not changed:
                break
                
        return levels
    

    # Returns a grammar with no actual recursive symbols: each rule only
    # contains the sequences of terminal symbols (the ones in the original sequence)
    def get_expanded_grammar(self):

        expanded_grammar = {}

        # Memoization cache to store rules we have already expanded
        memo = {}

        def _expand_recursive(symbol):

            # Check if we have already calculated this symbol
            if symbol in memo:
                return memo[symbol]

            # If the symbol is not a key in the grammar, it is a terminal
            # symbol (or at least cannot be expanded further).
            if symbol not in self.grammar:
                return (symbol,)

            # Otherwise if the symbol is a rule iterate through its
            # components and expand them.
            full_expansion = []
            for component in self.grammar[symbol]:
                full_expansion.extend(_expand_recursive(component))
            
            # Convert list to tuple for immutability and consistency
            result = tuple(full_expansion)
            
            # Save to cache
            memo[symbol] = result
            return result

        # Iterate over all rules currently defined in the grammar
        for rule_name in self.grammar:
            expanded_grammar[rule_name] = _expand_recursive(rule_name)

        return expanded_grammar


    def plot_grammar(self, ax=None, fig=None, original_symbols_order=None, 
                              node_color_mode='count', node_size=100, node_spacing=2.0, node_shape='s',
                              label_color='k', cmap=None, fontsize=12, reverse_arrows=True):

        # Create the graph
        graph = self.get_grammar_graph()

        # Compute the depth of each node
        rule_depth = self.get_grammar_depth(graph)

        # Init the position dict
        pos = {}

        # Place level 0 symbols
        # If the order is not given, go randomly
        if original_symbols_order is not None:
            for i, node in enumerate(original_symbols_order):
                if node in graph:
                    pos[node] = (i * node_spacing, 0)
        else:
            terminal_symbols = [r for r in rule_depth if rule_depth[r] == 0]
            for i, node in enumerate(terminal_symbols):
                pos[node] = (i * node_spacing, 0)

        # Group all nodes based on their level
        nodes_by_level = {}
        for n, l in rule_depth.items():
            if l not in nodes_by_level: nodes_by_level[l] = []
            nodes_by_level[l].append(n)
        max_level = max(rule_depth.values())

        # Loop over levels (bottom to top, except first)
        for l in range(1, max_level + 1):
            level_nodes = nodes_by_level.get(l, [])
            
            # Calculate ideal x position (average of children's x)
            node_data = []
            for node in level_nodes:
                children = list(graph.successors(node))
                if children:
                    avg_x = np.mean([pos[c][0] for c in children if c in pos])
                else:
                    avg_x = 0
                node_data.append((node, avg_x))
            
            # Sort by ideal x to determine relative order
            node_data.sort(key=lambda x: x[1])
            
            # Assign actual x, preventing overlap
            if not node_data: continue
            
            # Start placing nodes. 
            # To keep them centered, we can use the ideal x as a starting point,
            # but push them right if they overlap with the previous node.
            
            # Initial placement
            current_xs = [x[1] for x in node_data]
            
            # Resolve overlaps (sweep left-to-right)
            for i in range(1, len(current_xs)):
                if current_xs[i] < current_xs[i-1] + node_spacing:
                    current_xs[i] = current_xs[i-1] + node_spacing
                    
            # Assign to pos
            for (node, _), x in zip(node_data, current_xs):
                pos[node] = (x, l * 2.0) # y = Level * vertical_spacing

        nodes_positions = pos


        # If we need to use the symbol count as a color, compute them
        # Note that we still need to include the terminal symbols which may have a count of 0
        if node_color_mode == 'count':
            symbol_counts = []
            seq_symbols_count = Counter(self.sequence)
            for node in graph.nodes:
                if node in seq_symbols_count:
                    symbol_counts.append(seq_symbols_count[node])
                else:
                    symbol_counts.append(0)

        elif node_color_mode == 'recursive_count':
            symbol_counts = []
            seq_symbols_count = self.get_recursive_counts()
            for node in graph.nodes:
                if node in seq_symbols_count:
                    symbol_counts.append(seq_symbols_count[node])
                else:
                    symbol_counts.append(0)         
            
        # Change the labels to make them more readable (this is hard-coded now, need to pass a function)
        # labels = graph.nodes.keys()
        # label_rename_dict = {l : '\n'.join(l.split('_')) for l in labels}
        # nx.relabel_nodes(graph, label_rename_dict, copy=False)
        # nodes_positions = {label_rename_dict[l]: nodes_positions[l] for l in nodes_positions}
        
        # Create the figure if needed
        if fig is None:
            fig = plt.figure(figsize=(15,15))
        if ax is None:
            ax = fig.add_subplot()

        if node_color_mode == 'count':
            node_colors = symbol_counts
        elif node_color_mode == 'recursive_count':
            node_colors = np.log(1 + np.array(symbol_counts))
        else:
            node_colors = node_color_mode

        if cmap is None:
            cmap = 'viridis'

        # Reverse the arrows if needed for visualization
        if reverse_arrows:
            graph = nx.DiGraph.reverse(graph)

        # Draw nodes
        nx.draw_networkx_nodes(graph, nodes_positions, node_color=node_colors, node_shape=node_shape, cmap=cmap, node_size=node_size, alpha=1, edgecolors='k')
        
        # Draw edges
        nx.draw_networkx_edges(graph, nodes_positions, edge_color='k', alpha=1, node_size=node_size, arrows=True, arrowsize=node_size/100)

        # Draw labels
        nx.draw_networkx_labels(graph, nodes_positions, font_size=fontsize, font_weight='bold', font_color=label_color)            

        plt.axis('off')


        # Show the colorbar
        if len(symbol_counts) > 0:
            norm=mpl.colors.Normalize(np.min(symbol_counts), np.max(symbol_counts))
                                
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap=cmap), ax=ax,
                        orientation='vertical')
            cb.set_label(label='Occurrences', fontsize=18)

        plt.tight_layout()





    def summary(self):
        print('-'*80)
        print(f'{"":<30}VOCABULARY')
        print('-'*80)
        print(f'Started from a vocabulary of {len(self.vocabulary_init)} symbols')
        print(f'Final vocabulary contains {len(self.vocabulary)} symbols')
        print('-'*80)
        print(f'{"":<30}SEQUENCE')
        print('-'*80)
        print(f'Started from the sequence:')
        print(self.sequence_init)
        print(f'Of length {len(self.sequence_init)} and containing {len(np.unique(self.sequence_init))} unique symbols')
        print(f'Ended with the sequence:')
        print(self.sequence)
        print(f'Of length {len(self.sequence)} and containing {len(np.unique(self.sequence))} unique symbols')
        print('-'*80)
        print(f'{"":<30}GRAMMAR')
        print('-'*80)
        print(f'The final grammar is:')
        print(self.grammar)
        print(f'Containing {len(self.grammar)} rules')
        print('-'*80)
        print(f'During the fit procedure we pruned {self.rules_pruned_num} rules')
