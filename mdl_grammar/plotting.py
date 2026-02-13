import numpy as np
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from mdl_grammar.grammar_models import GrammarModel


# Instead of plotting the specific symbol we replace it with the whole
# sequence of terminal symbols (in the right order)
def plot_grammar_expanded(grammar_model: GrammarModel, ax=None, fig=None,
                            node_size=1, node_spacing=0.2, vertical_spacing=2.0,
                            cmap=None, color_mode='symbol', original_symbols_order=None,
                            label_color='w', fontsize=12, reverse_arrows=True, pretty_label_fn=None,
                            colorbar_scale=1.0):

    # Create the graph
    graph = grammar_model.get_grammar_graph()

    # Compute the depth of each node
    rule_depth = grammar_model.get_grammar_depth(graph)

    # Compute the expanded grammar (only terminal symbols)
    expanded_grammar = grammar_model.get_expanded_grammar()

    # Also create a list of the terminal symbols (excluding separators)
    terminal_sym = [r for r in rule_depth if rule_depth[r] == 0]

    # Given the expanded grammar compute the size of each node (except the terminal ones)
    nodes_set_len = {rule: len(expanded_grammar[rule]) for rule in expanded_grammar}

    # Count the occurrences of all symbols in the final sequence
    symbol_counts = grammar_model.get_counts()
    max_count = max(symbol_counts.values())
    min_count = min(symbol_counts.values())
    symbol_counts_normalized = {s: (symbol_counts[s] - min_count)/(max_count - min_count) for s in symbol_counts}

    # Init the position dict
    pos = {}

    # Place level 0 symbols. If the order is not given, go randomly
    if original_symbols_order is not None:
        for i, node in enumerate(original_symbols_order):
            if node in graph:
                pos[node] = (i * node_spacing - len(terminal_sym) // 2 * node_spacing, 0)
    else:
        for i, node in enumerate(terminal_sym):
            pos[node] = (i * node_spacing - len(terminal_sym) // 2 * node_spacing, 0)

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
        
        if not node_data: continue
        
        # Start placing nodes. 
        # To keep them centered, we can use the ideal x as a starting point,
        # but push them right if they overlap with the previous node.
        
        # Initial placement
        current_xs = [x[1] for x in node_data]

        # Compute the size of the boxes in this layer (add 1 to the length to accont for padding)
        current_sizes = [(nodes_set_len[x[0]] + 0.5) * node_spacing * 1.1 for x in node_data]
        
        # Resolve overlaps (sweep left-to-right)
        # Make sure to use the actual estimated size
        for i in range(1, len(current_xs)):
            
            # Left border of this box
            x_left_current = current_xs[i] - current_sizes[i]/2

            # Right border of the previous box
            x_right_prev = current_xs[i-1] + current_sizes[i-1]/2

            if x_left_current < x_right_prev:
                current_xs[i] = x_right_prev + current_sizes[i]/2

        # Remove the average X to center this layer
        # current_xs = np.array(current_xs)
        # current_xs = current_xs - current_xs.mean()

        # Assign to pos
        for (node, _), x in zip(node_data, current_xs):
            pos[node] = (x, l * vertical_spacing) # y = Level * vertical_spacing

    nodes_positions = pos

    # Color management for the terminal nodes
    if cmap is None:
        cmap = 'viridis'
    colormap = plt.get_cmap(cmap)

    if color_mode == 'symbol':
        # Each symbol gets its own color
        terminal_node_colors = {sym: colormap(i/(len(terminal_sym)-1)) for i, sym in enumerate(terminal_sym)}
    elif color_mode == 'counts':
        # Color based on the frequency (in the final sequence)
        terminal_node_colors = {sym: colormap(symbol_counts_normalized[sym]) for sym in terminal_sym}



    # Create the figure if needed
    if fig is None:
        x_pos = [nodes_positions[p][0] for p in nodes_positions]
        pos_delta = max(x_pos) - min(x_pos)
        figsize = (pos_delta, max_level / 3 * 5)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
    elif ax is None:
        ax = fig.add_subplot()

    if pretty_label_fn is None:
        pretty_label_fn = lambda s: s
    
    # To store center positions and heights of the various components (used to draw arrows)
    nodes_centers = {}
    nodes_heights = {}

    # First, plot the terminal nodes
    for node in terminal_sym:
        center_pos = nodes_positions[node]
        corner_pos = (center_pos[0] - node_size/2, center_pos[1] - node_size/2)
        box = Rectangle(corner_pos, node_size, node_size, color=terminal_node_colors[node])
        ax.add_patch(box)
        ax.text(center_pos[0], center_pos[1], pretty_label_fn(node), ha='center', va='center', color=label_color, fontsize=fontsize, fontweight='bold')

        nodes_centers[node] = center_pos
        nodes_heights[node] = node_size

    # Plot the boxes for higher levels
    for node in nodes_positions:
        if node not in terminal_sym:

            # Big symbol box
            width = (nodes_set_len[node] + 0.5) * node_spacing
            height = node_size * 1.5
            center_pos = nodes_positions[node]
            corner_pos = (center_pos[0] - width/2, center_pos[1] - height/2)
            box = FancyBboxPatch(corner_pos, width, height, boxstyle='Round, pad=0, rounding_size=0.2',
                                 alpha=0.3, linewidth=0, color='gray')
            ax.add_patch(box)
            ax.text(center_pos[0], center_pos[1] + height/2*1.25, node, ha='center', va='bottom',
                    color='k', fontsize=fontsize, fontweight='bold', backgroundcolor='w')
            nodes_centers[node] = center_pos
            nodes_heights[node] = height

            # Then all the terminal boxes inside
            subnodes = expanded_grammar[node]

            for i in range(len(subnodes)):
                sym = subnodes[i]
                c_x = i * node_spacing + center_pos[0] - (nodes_set_len[node]-1) * node_spacing/2
                c_y = center_pos[1]

                # The color of the box depend on the color_mode. If we are coloring by count we must use the parent symbol count
                if color_mode == 'symbol':
                    color = terminal_node_colors[sym]
                elif color_mode == 'counts':
                    color = colormap(symbol_counts_normalized[node])

                box = Rectangle((c_x - node_size/2, c_y - node_size/2), node_size, node_size, color=color)
                ax.add_patch(box)     
                ax.text(c_x, c_y, pretty_label_fn(sym), ha='center', va='center', color=label_color, 
                        fontsize=fontsize, fontweight='bold')
        
    
    # Draw the arrows
    for edge in graph.edges:

        node_start = edge[0]
        node_end = edge[1]

        if reverse_arrows:
            node_start, node_end = node_end, node_start

        # Use the centers and heights to determine the start and end positions
        x_start = nodes_centers[node_start][0]
        y_start = nodes_centers[node_start][1] 
        x_end = nodes_centers[node_end][0]
        y_end = nodes_centers[node_end][1]        

        # Used to correctly compute the anchor position 
        dir = 1 if y_end > y_start else -1

        y_start += nodes_heights[node_start] * dir / 2
        y_end -= nodes_heights[node_end] * dir / 2

        # Draw the arrow
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                color=(0.1,0.1,0.1), zorder=-1, alpha=0.2,
                                arrowstyle='simple, head_length=3, head_width=3, tail_width=0.05')
        ax.add_patch(arrow)


    # Show the colorbar
    if color_mode == 'counts':
        norm=mpl.colors.Normalize(min_count, max_count)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap=cmap), ax=ax,
                    orientation='vertical', aspect=10, fraction=0.15, shrink=0.5 * colorbar_scale)
        cb.set_label(label='Occurrences', fontsize=18)


    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()