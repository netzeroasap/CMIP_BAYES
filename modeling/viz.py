from graphviz import Digraph

def show_highlevel_tree(kind="jupyter", save_filename=None):
    """
    Show the high-level ESM decision tree.
    
    Parameters
    ----------
    kind : str
        One of ["talk", "paper", "jupyter"], affects layout and styling.
    save_filename : str or None
        If provided, save the graph to this filename (extension determines format).
    """
    
    # --- Graph defaults ---
    if kind == "talk":
        graph_attr = {
            "rankdir": "LR",         # landscape for talk
            "bgcolor": "white",
            "fontsize": "14",
            "nodesep": "0.5",
            "ranksep": "1.0"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "14",
            "width": "3.5"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "12"
        }
    elif kind == "paper":
        graph_attr = {
            "rankdir": "TB",        # vertical for paper
            "bgcolor": "white",
            "fontsize": "10",
            "nodesep": "0.35",
            "ranksep": "0.5"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "10.5",
            "width": "3.0"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "9.5"
        }
    elif kind == "jupyter":
        graph_attr = {
            "rankdir": "TB",
            "bgcolor": "white",
            "fontsize": "12",
            "nodesep": "0.4",
            "ranksep": "0.6"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "12",
            "width": "3.0"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "10"
        }
    else:
        raise ValueError("kind must be one of ['talk','paper','jupyter']")
    
    # --- Create graph ---
    dot = Digraph(
        engine="dot",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr
    )
    
    # --- Nodes ---
    dot.node("A", "Do ESMs measure the same underlying X?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("B", "Use fixed-effects model", fillcolor="#E8F5E9")
    dot.node("C", "Are there shared sources of bias across ESMs?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("P", "Known biases from identifiable processes?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("S", "Model structured bias (additive/multiplicative)\n+ residual random effects", fillcolor="#E8F5E9")
    dot.node("D", "Use uncorrelated random-effects model\n(ρ = 0)", fillcolor="#E8F5E9")
    dot.node("E", "How much of the residual bias is shared?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("F", "Fix residual correlation\nρ = 0.5", fillcolor="#E8F5E9")
    dot.node("G", "Estimate residual correlation\nρ ~ Beta(5,5)", fillcolor="#E8F5E9")
    dot.node("SIGMA", "Decide σᵢ priors\n(for each ensemble member)\n→ see σᵢ tree", shape="note", fillcolor="#FFF0F5")
    
    # --- Edges ---
    dot.edge("A", "B", label="Yes")
    dot.edge("A", "C", label="No")
    
    dot.edge("C", "D", label="No")
    dot.edge("C", "P", label="Yes")
    
    dot.edge("P", "S", label="Yes")
    dot.edge("P", "E", label="No")
    
    dot.edge("S", "E", label="Residual bias")
    dot.edge("E", "F", label="Known (~50%)")
    dot.edge("E", "G", label="Unknown")
    
    # Link residual RE nodes to σᵢ tree
    for node in ["D", "F", "G"]:
        dot.edge(node, "SIGMA", style="dashed", label="next step")
    
    # --- Save if requested ---
    if save_filename is not None:
        dot.render(save_filename, view=False, cleanup=True)
    
    return dot




def show_iv_tree(kind="jupyter", save_filename=None):
    """
    Show the internal variability (σᵢᵥ) decision tree.
    
    Parameters
    ----------
    kind : str
        One of ["talk","paper","jupyter"], affects layout and styling.
    save_filename : str or None
        If provided, save the graph to this filename (extension determines format).
    """
    
    # --- Graph defaults ---
    if kind == "talk":
        graph_attr = {
            "rankdir": "LR",  # landscape for talk
            "bgcolor": "white",
            "fontsize": "14",
            "nodesep": "0.5",
            "ranksep": "1.0"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "14",
            "width": "4.0"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "12"
        }
    elif kind == "paper":
        graph_attr = {
            "rankdir": "TB",
            "bgcolor": "white",
            "fontsize": "10",
            "nodesep": "0.35",
            "ranksep": "0.5"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "11",
            "width": "3.0"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "10"
        }
    elif kind == "jupyter":
        graph_attr = {
            "rankdir": "TB",
            "bgcolor": "white",
            "fontsize": "12",
            "nodesep": "0.4",
            "ranksep": "0.6"
        }
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#F8F9FA",
            "color": "#333333",
            "fontname": "Helvetica",
            "fontsize": "12",
            "width": "3.0"
        }
        edge_attr = {
            "color": "#555555",
            "fontname": "Helvetica",
            "fontsize": "10"
        }
    else:
        raise ValueError("kind must be one of ['talk','paper','jupyter']")
    
    # --- Create graph ---
    dot = Digraph(
        engine="dot",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr
    )
    
    # --- Nodes ---
    dot.node("A", "More than one\nensemble member?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("B", "Ensemble size?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("C", "Is internal\nvariability negligible?", shape="diamond", fillcolor="#FFF3CD")
    
    dot.node("D", "Medium–large ensemble\n\n• Estimate σᵢᵥ from data\n• Wide prior ~ N⁺(.)\n  for each ESM", fillcolor="#E8F5E9")
    dot.node("E", "Few members\n\n• Data weakly informs σᵢᵥ\n• Same narrow informative\n  prior ~ N⁺(.) for all ESMs", fillcolor="#E8F5E9")
    dot.node("F", "σᵢᵥ negligible\n\n• X known almost exactly\n• Fix σᵢᵥ to small constant", fillcolor="#E3F2FD")
    dot.node("G", "σᵢᵥ non-negligible\n\n• Prior provides all information\n• Choose μᵢᵥ\n• Marginalize IG prior\n• Xᵢ ~ Student-t(5, μᵢᵥ)\n  (fat tails = extra uncertainty)", fillcolor="#E3F2FD")
    
    # --- Edges ---
    dot.edge("A", "B", label="Yes")
    dot.edge("A", "C", label="No")
    
    dot.edge("B", "D", label="Medium–large")
    dot.edge("B", "E", label="Small")
    
    dot.edge("C", "F", label="Yes")
    dot.edge("C", "G", label="No")
    
    # --- Save if requested ---
    if save_filename is not None:
        dot.render(save_filename, view=False, cleanup=True)
    
    return dot

