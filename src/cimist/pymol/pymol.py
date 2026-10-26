import os
import json
import numpy as np
import matplotlib as mpl
import cmocean as cmo
import cmasher as cm
import mdtraj as md
import matplotlib.pyplot as plt
import dill as pkl

CMAPS = {k : v for k,v in cmo.cm.__dict__.items() if "LinearSegmentedColormap" in str(type(v))}
CMAPS.update({k : v for k,v in cm.__dict__ .items() if "ListedColormap" in str(type(v))})


def _reference_ca_indices(tree, topology):
    """Resolve explicit model identities; legacy trees retain topology ordering."""
    records = [tree.nodes[node] for node in tree.nodes()]
    if all("ca_atom_index" in record for record in records):
        indices = []
        for record in records:
            index = int(record["ca_atom_index"])
            if not 0 <= index < topology.n_atoms:
                raise ValueError("Reference topology differs from the fitted topology")
            atom = topology.atom(index)
            residue = atom.residue
            # The generated PDB truncates chain IDs/resSeq and names blank chains.
            pdb_chain = (record["chain_id"][:1] if record["chain_id"] else
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[record["chain_index"] % 26])
            if (atom.name != "CA" or residue.index != record["residue_index"]
                    or residue.chain.index != record["chain_index"]
                    or (residue.chain.chain_id or "") not in (record["chain_id"], pdb_chain)
                    or residue.resSeq not in (record["resSeq"], record["resSeq"] % 10000)
                    or residue.name != record["name"]):
                raise ValueError("Reference topology differs from the fitted residue identities")
            indices.append(index)
        return np.asarray(indices)
    if any("ca_atom_index" in record for record in records):
        raise ValueError("Tree contains incomplete residue identities")
    indices = topology.select("protein and name CA")
    if len(indices) != len(records):
        raise ValueError("Reference must contain one protein CA atom per tree residue")
    return indices


def tree_cartoon(tree, protein_structure, savedir,
                 edge_cmap="turbid", edge_vmin=0, edge_vmax=1,
                 node_vmin=0, node_vmax=3.5, min_MI=0.2, base_stick_radius=1,
                base_sphere_radius=1, entropy_decay_factor=1, node_cmap="algae",
                cbar_figsize=(3,1),
                min_stick_radius=0.1, *, focus_nodes=None):
    """Export the inferred tree or a display-only focus on its existing paths.

    A focus retains excluded connector nodes and full-model statistics. The
    selected Eq. 3 allocations in focus.json are shares of full-model entropy.
    Use refit_to_subset to infer a marginal subsystem instead.
    """
    savedir = os.fspath(savedir)
    graph = tree.T if focus_nodes is None else tree.trim(focus_nodes)
    reslist = list(graph.nodes())
    protein_structure = protein_structure.slice(0, copy=False)
    full_ca_indices = _reference_ca_indices(tree, protein_structure.topology)
    ca_by_node = dict(zip(tree.nodes(), full_ca_indices))
    ca_indices = np.asarray([ca_by_node[node] for node in reslist])

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if type(edge_cmap) == str:
        edge_cmap = CMAPS[edge_cmap]
    
    bfactors = np.exp(-entropy_decay_factor * tree.residue_entropies().loc[reslist].values)
    bfactors_structure = np.zeros(protein_structure.xyz.shape[1])
    bfactors_structure[ca_indices] = bfactors
    protein_structure.save_pdb(savedir + os.sep + "structure.pdb", bfactors=bfactors_structure)
    
    # CREATE PDB FILE WITH ONLY ALPHA CARBONS
    ca_topology = md.Topology()
    previous_chain = None
    for i, index in enumerate(ca_indices):
        original = protein_structure.topology.atom(int(index)).residue
        if original.chain is not previous_chain:
            chain = ca_topology.add_chain(chain_id=original.chain.chain_id)
            previous_chain = original.chain
        residue = ca_topology.add_residue(original.name, chain, original.resSeq,
                                         original.segment_id)
        ca_topology.add_atom("CA", md.element.carbon, residue, serial=i + 1)
    CA_only = md.Trajectory(protein_structure.xyz[:, ca_indices], ca_topology)
    
    
    
    # SAVE TREE EDGES AS CONNECT RECORDS
    connect_records = []
    
    bond_creation_lines = [5*"\n" + "#MAKE BONDS FOR TREE EDGES\n"]
    set_bond_radius_lines = [5*"\n" + "#TREE EDGE RADIUS SETTINGS\n"]
    norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    
    set_color_lines = [5*"\n", "#COLOR DEFINITIONS\n"]
    set_bond_color_lines = [5*"\n", "#BOND COLOR SETTINGS\n"]
    
    set_q_lines = [5*"\n", "#q-values set to entropy for spectrum\n"]
    
    for (u,v) in graph.edges():
        
        # atom indices
        a1 = reslist.index(u)+1
        a2 = reslist.index(v)+1
        #connect_records.append(f"CONECT {a1} {a2}\n")
        
        # residue numbers
        #r1 = CA_only_df.loc[reslist.index(u), "resSeq"]
        #r2 = CA_only_df.loc[reslist.index(v), "resSeq"]
        bond_creation_lines.append(
            f"cmd.bond('tree and index {a1}', 'tree and index {a2}')\n"
        )
        # define color for tree edge
        color_name = f"I_{a1}_{a2}"
        I = graph.edges[u,v][tree._I_key]
        RGB = edge_cmap(norm(I))[:-1]
        RGB = tuple(np.format_float_positional(x, 16) for x in RGB)
                        
        set_bond_color_lines.append(
            f"cmd.set_bond('stick_color', '{color_name}', 'tree and index {a1}', 'tree and index {a2}')\n"
        )
        
        distance = np.squeeze(md.compute_distances(CA_only, [[a1-1, a2-1]]))
        if I > min_MI:
            radius = base_sphere_radius * np.sqrt(I / (np.pi * distance))
        else:
            radius = 0
        set_bond_radius_lines.append(
            f"cmd.set_bond('stick_radius', {max(radius, min_stick_radius)}, 'tree and index {a1}', 'tree and index {a2}')\n")
        #set_bond_color_lines.append(
        #    f"cmd.set_bond('stick_color', '{color_name}', 'tree and id {a2}', 'tree and id {a1}')\n"
        #)
        
        set_color_lines.append(f"cmd.set_color('{color_name}', {RGB})\n")
        
    for i, (n, index) in enumerate(zip(reslist, ca_indices), start=1):
        entropy_ = graph.nodes[n][tree._S_key]
        set_q_lines.append(
            f"cmd.alter('structure and index {int(index) + 1}', 'q={entropy_}')\n")
        set_q_lines.append(
            f"cmd.alter('tree and index {i}', 'q={entropy_}')\n")
        
        set_q_lines.append(
            f"cmd.alter('tree and index {i}', 'vdw={base_sphere_radius*float(np.power(entropy_, 1/3))}')\n")
        
    tree_file = f"{savedir}" + os.sep + "tree.pdb"
    CA_only.save_pdb(tree_file, bfactors=bfactors)
    
    with open(tree_file, "r") as f:
        pdblines = f.readlines()[:-1]
    
    
  
    pmlfname = savedir + os.sep + "draw_tree.pml"
    with open(pmlfname, "w") as f:
        load_lines = ["set retain_order, 1\n", "load structure.pdb\n", "load tree.pdb\n", "hide cartoon, tree"]
        
        pml_lines =  load_lines + bond_creation_lines + set_color_lines + set_bond_color_lines + set_bond_radius_lines
        pml_lines += set_q_lines
        pml_lines += ["set cartoon_cylindrical_helices, 1\n",
                      f"spectrum q, {node_cmap}, name CA, {node_vmin}, {node_vmax}\n"]
        f.writelines(pml_lines + ["show sticks, tree\n",
                                  "show spheres, tree\n",
                                  "set cartoon_transparency, 0.6\n",
                                  "set sphere_transparency, 0.3, all\n",
                                  "set cartoon_putty_radius, 0.25\n",
                                  "set cartoon_putty_transform, 4\n",
                                  "set cartoon_putty_scale_power, 1.\n",
                                  "set cartoon_putty_scale_min, -1\n",
                                  "set cartoon_putty_scale_max, -1\n",
                                  "orient\n",
                                  "zoom visible, complete=1\n"],
                    )

        if focus_nodes is not None:
            f.write("\n# Full-model focus view. Eq. 3 entropy allocations are in focus.json.\n")
            for selection, selected in (("focus_residues", True), ("connector_residues", False)):
                atoms = [str(i) for i, node in enumerate(reslist, 1)
                         if graph.nodes[node]["is_selected"] == selected]
                expression = "tree and index " + "+".join(atoms) if atoms else "none"
                f.write(f"select {selection}, {expression}\n")
            f.write("set sphere_transparency, 0.65, connector_residues\n")

    focus_path = savedir + os.sep + "focus.json"
    if focus_nodes is not None:
        records = []
        identity_fields = ("residue_index", "chain_index", "chain_id", "resSeq",
                           "name", "segment_id", "ca_atom_index")
        for i, (node, atom_index) in enumerate(zip(reslist, ca_indices), 1):
            attrs = graph.nodes[node]
            identity = {key: attrs[key] for key in identity_fields if key in attrs}
            records.append(dict(label=node, tree_atom_index_1based=i,
                reference_atom_index_0based=int(atom_index),
                is_selected=attrs["is_selected"], identity=identity,
                full_model_entropy_contribution_nats=attrs["full_model_entropy_contribution"]))
        metadata = dict(view_kind="full_tree_paths", semantics="display_only",
            entropy_allocation="full_model_eq3", prior=tree.prior,
            entropy_estimator=tree.entropy_estimator, N_obs=int(tree.N_obs),
            model_selection=tree.T.graph.get("residue_selection"),
            full_model_node_count=len(tree.nodes),
            selected_full_model_entropy_contribution_nats=sum(
                record["full_model_entropy_contribution_nats"] for record in records if record["is_selected"]),
            residues=records)
        with open(focus_path, "w") as f:
            json.dump(metadata, f, indent=2, allow_nan=False, default=np.generic.item)
    elif os.path.exists(focus_path):
        # Reusing a focus directory for a complete view must clear its old label.
        os.remove(focus_path)

    # CREATE THE COLORBAR
    fig, ax = plt.subplots(2, 1, figsize=cbar_figsize, layout='constrained')
    norm_mi = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    norm_s = mpl.colors.Normalize(vmin=node_vmin, vmax=node_vmax)

    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_mi, cmap=edge_cmap),
                cax=ax[0], orientation='horizontal')
    ax[0].xaxis.set_ticks_position("top")
    ax[0].xaxis.set_label_position("top")

    ax[0].set_xticks([edge_vmin, edge_vmax])
    ax[0].set_xlabel('Mutual Information (nats)', labelpad=-10, fontsize=7)

    node_cmap = CMAPS[node_cmap]
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_s, cmap=node_cmap),
                cax=ax[1], orientation='horizontal')
    ax[1].set_xticks([node_vmin, node_vmax])

    ax[1].set_xlabel('Entropy (nats)', labelpad=-10, fontsize=7)
    fig.savefig(f"{savedir}/colorbar.png", dpi=300)

    

    # PICKLE THE COLORBAR FOR LATER EDITING IF NEEDED
    with open(f"{savedir}" + os.sep + "colorbar.pkl", "wb") as f:
        pkl.dump((fig, ax), f)
    
    print(f"Output files written to directory {savedir}.")
    print(f"Navigate to {savedir} and run 'pymol draw_tree.pml' to view tree and open 'colorbar.png' to view colorbar.")

    return fig, ax
