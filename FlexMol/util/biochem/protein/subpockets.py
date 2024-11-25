
import dgl
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
import deepchem
import numpy as np
import os
import pandas as pd
import torch
import dgl.backend as F

#TODO clean this file

pk = deepchem.dock.ConvexHullPocketFinder()

def merge(graphs):
    r"""Merge a sequence of graphs together into a single graph.

    Nodes and edges that exist in ``graphs[i+1]`` but not in ``dgl.merge(graphs[0:i+1])``
    will be added to ``dgl.merge(graphs[0:i+1])`` along with their data.
    Nodes that exist in both ``dgl.merge(graphs[0:i+1])`` and ``graphs[i+1]``
    will be updated with ``graphs[i+1]``'s data if they do not match.

    Parameters
    ----------
    graphs : list[DGLGraph]
        Input graphs.

    Returns
    -------
    DGLGraph
        The merged graph.

    Notes
    ----------
    * Inplace updates are applied to a new, empty graph.
    * Features that exist in ``dgl.graphs[i+1]`` will be created in
      ``dgl.merge(dgl.graphs[i+1])`` if they do not already exist.

    Examples
    ----------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0,1]), torch.tensor([2,3])))
    >>> g.ndata["x"] = torch.zeros(4)
    >>> h = dgl.graph((torch.tensor([1,2]), torch.tensor([0,4])))
    >>> h.ndata["x"] = torch.ones(5)
    >>> m = dgl.merge([g, h])

    ``m`` now contains edges and nodes from ``h`` and ``g``.

    >>> m.edges()
    (tensor([0, 1, 1, 2]), tensor([2, 3, 0, 4]))
    >>> m.nodes()
    tensor([0, 1, 2, 3, 4])

    ``g``'s data has updated with ``h``'s in ``m``.

    >>> m.ndata["x"]
    tensor([1., 1., 1., 1., 1.])

    See Also
    ----------
    add_nodes
    add_edges
    """

    ref = graphs[0]
    ntypes = ref.ntypes
    etypes = ref.canonical_etypes
    data_dict = {etype: ([], []) for etype in etypes}
    num_nodes_dict = {ntype: 0 for ntype in ntypes}
    merged = dgl.heterograph(data_dict, num_nodes_dict, ref.idtype, ref.device)

    # Merge edges and edge data.
    for etype in etypes:
        unmerged_us = []
        unmerged_vs = []
        edata_frames = []
        for graph in graphs:
            etype_id = graph.get_etype_id(etype)
            us, vs = graph.edges(etype=etype)
            unmerged_us.append(us)
            unmerged_vs.append(vs)
            edge_data = graph._edge_frames[etype_id]
            edata_frames.append(edge_data)
        keys = ref.edges[etype].data.keys()
        if len(keys) == 0:
            edges_data = None
        else:
            edges_data = {k: F.cat([f[k] for f in edata_frames], dim=0) for k in keys}
        merged_us = F.copy_to(F.astype(F.cat(unmerged_us, dim=0), ref.idtype), ref.device)
        merged_vs = F.copy_to(F.astype(F.cat(unmerged_vs, dim=0), ref.idtype), ref.device)
        merged.add_edges(merged_us, merged_vs, edges_data, etype)

    # Add node data and isolated nodes from next_graph to merged.
    for next_graph in graphs:
        for ntype in ntypes:
            merged_ntype_id = merged.get_ntype_id(ntype)
            next_ntype_id = next_graph.get_ntype_id(ntype)
            next_ndata = next_graph._node_frames[next_ntype_id]
            node_diff = (next_graph.num_nodes(ntype=ntype) -
                         merged.num_nodes(ntype=ntype))
            n_extra_nodes = max(0, node_diff)
            merged.add_nodes(n_extra_nodes, ntype=ntype)
            next_nodes = F.arange(
                0, next_graph.num_nodes(ntype=ntype), merged.idtype, merged.device
            )
            merged._node_frames[merged_ntype_id].update_row(
                next_nodes, next_ndata
            )

    return merged

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 6, 5, 6, 1) --> total 28


def get_atom_feature(atoms):
    H = []
    for atom in atoms:
        H.append(atom_feature(atom))
    H = np.array(H)
    return H




def calculate_bounding_box(molecule):
    conformer = molecule.GetConformer()
    positions = conformer.GetPositions()
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    return min_coords, max_coords

def atoms_within_box(molecule, box_min, box_max):
    conformer = molecule.GetConformer()
    positions = conformer.GetPositions()
    indices = []
    for idx, pos in enumerate(positions):
        if np.all(pos >= box_min) and np.all(pos <= box_max):
            indices.append(idx)
    return indices


def parse_pdb_for_bounding_box(pdb_file):
    with open(pdb_file, 'r') as file:
        x_coords, y_coords, z_coords = [], [], []
        for line in file:
            if line.startswith("ATOM"):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
        min_coords = np.array([min(x_coords), min(y_coords), min(z_coords)])
        max_coords = np.array([max(x_coords), max(y_coords), max(z_coords)])
    return min_coords, max_coords



def process_protein_subpocket(protein_name, pdb_dir, subpocket_dir, max_subpockets=30):
    # data_dir = f"data/BIOSNAP/subpocket/long/"
    long_data_dir = subpocket_dir + "long/"
    csv_file_path = os.path.join(long_data_dir, f"{protein_name}.csv")
    if os.path.exists(csv_file_path):
        print("long!")
        constructed_graphs = []
        df = pd.read_csv(csv_file_path)
        all_pdb = set(df["segment_folder_name"])
        complete_pdb_file_dict = {}
        am_dict = {}

        for pdb_name in all_pdb:
            pdb_path = f"{pdb_dir}{protein_name}/{pdb_name}.pdb"
            complete_pdb_file_dict[pdb_name] = Chem.MolFromPDBFile(pdb_path)
            if complete_pdb_file_dict[pdb_name] is None:
                raise ValueError(f"Could not read molecule from {pdb_path}")
            am_dict[pdb_name] = GetAdjacencyMatrix(complete_pdb_file_dict[pdb_name])

        subpocket_count = 0
        pocket_number = 1

        for index, row in df.iterrows():
            if pocket_number >= 10: 
                break
            if subpocket_count >= max_subpockets:
                break
            folder = row["segment_folder_name"]
            base_dir = f"{long_data_dir}{protein_name}/{folder}/"
            pocket_index = row["pocket_index"]
            subpocket_number = 1
            batched_graphs = []
            while True:
                subpocket_file = os.path.join(base_dir, f"subpocket_{pocket_index}_{subpocket_number}.pdb")
                if subpocket_count >= max_subpockets:
                    break
                if not os.path.exists(subpocket_file):
                    break
                else:
                    print(subpocket_file)
                    complete_molecule = complete_pdb_file_dict[folder]
                    am = am_dict[folder]
                    box_min, box_max = parse_pdb_for_bounding_box(subpocket_file)
                    atom_indices = atoms_within_box(complete_molecule, box_min, box_max)
                    ami = am[np.ix_(atom_indices, atom_indices)]
                    g = nx.from_numpy_array(ami)
                    for component in list(nx.connected_components(g)):
                        if len(component) <=10:
                            g.remove_nodes_from(component)

                    remaining_node_indices = list(g.nodes)
                    selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                    H = get_atom_feature(selected_atoms)

                    graph = dgl.from_networkx(g)
                    graph.ndata['h'] = torch.Tensor(H)
                    graph = dgl.add_self_loop(graph)
                    graph.ndata['pocket_number'] = torch.full((graph.number_of_nodes(),), pocket_number, dtype=torch.float)

                    if graph.number_of_nodes() == 0:
                        subpocket_number += 1
                        continue

                    if subpocket_number <= 14:
                        constructed_graphs.append(graph)
                    elif subpocket_number > 20:
                        break
                    else:
                        batched_graphs.append(graph)

                    subpocket_number += 1
                    subpocket_count += 1

            if(len(batched_graphs) > 0):
                if(len(batched_graphs) == 1):
                    bg = batched_graphs[0]
                else:
                    bg = merge(batched_graphs)
                constructed_graphs.append(bg)

            pocket_number += 1

        while len(constructed_graphs) < max_subpockets:
            virtual_graph = dgl.graph(([], []))
            virtual_graph.add_nodes(1)
            virtual_graph.ndata['h'] = torch.zeros((1, 31))
            virtual_graph.ndata['pocket_number'] = torch.full((virtual_graph.number_of_nodes(),), -1, dtype=torch.float)
            constructed_graphs.append(virtual_graph)

        constructed_graphs = dgl.batch(constructed_graphs)
        return constructed_graphs

    else:
        complete_pdb_file = f"{pdb_dir}{protein_name}.pdb"
        complete_molecule = Chem.MolFromPDBFile(complete_pdb_file)
        if complete_molecule is None:
            complete_molecule = Chem.MolFromPDBFile(complete_pdb_file, sanitize = False)
            if complete_molecule is None:
                raise ValueError(f"Could not read molecule from {complete_pdb_file}")
        am = GetAdjacencyMatrix(complete_molecule)

        base_dir = f"{subpocket_dir}{protein_name}"
        constructed_graphs = []

        subpocket_count = 0
        pocket_number = 1
        exit = False
        while subpocket_count < max_subpockets and not exit:
            subpocket_number = 1
            batched_graphs = []
            while subpocket_number <= max_subpockets and subpocket_count < max_subpockets:
                subpocket_file = os.path.join(base_dir, f"subpocket_{pocket_number}_{subpocket_number}.pdb")
                if not os.path.exists(subpocket_file):
                    if subpocket_number == 1:
                        exit = True
                        break
                    break  # Go to the next pocket if the current subpocket doesn't exist

                box_min, box_max = parse_pdb_for_bounding_box(subpocket_file)
                atom_indices = atoms_within_box(complete_molecule, box_min, box_max)

                # H = get_atom_feature([complete_molecule.GetAtomWithIdx(idx) for idx in atom_indices])
                ami = am[np.ix_(atom_indices, atom_indices)]
                g = nx.from_numpy_array(ami)
                for component in list(nx.connected_components(g)):
                    if len(component) <=10:
                        g.remove_nodes_from(component)

                remaining_node_indices = list(g.nodes)
                selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                H = get_atom_feature(selected_atoms)

                graph = dgl.from_networkx(g)
                graph.ndata['h'] = torch.Tensor(H)
                graph = dgl.add_self_loop(graph)
                graph.ndata['pocket_number'] = torch.full((graph.number_of_nodes(),), pocket_number, dtype=torch.float)

                if graph.number_of_nodes() == 0:
                    subpocket_number += 1
                    continue
                if subpocket_number <= 14:
                    constructed_graphs.append(graph)
                elif subpocket_number > 20:
                    break
                else:
                    batched_graphs.append(graph)
                
                subpocket_count += 1
                subpocket_number += 1

            if(len(batched_graphs) > 0):
                if(len(batched_graphs) == 1):
                    bg = batched_graphs[0]
                else:
                    bg = merge(batched_graphs)
                constructed_graphs.append(bg)

            pocket_number += 1
            if pocket_number >= 10: 
                break
   


        if (len(constructed_graphs) == 0):
            print("No Subpockets Found! Use deepchem")
            
            pdb_file = f"{pdb_dir}/{protein_name}.pdb"
            m = Chem.MolFromPDBFile(pdb_file)
            if m is None:
                m = Chem.MolFromPDBFile(complete_pdb_file, sanitize = False)
            am = GetAdjacencyMatrix(m)
            pockets = pk.find_pockets(pdb_file)
            n2 = m.GetNumAtoms()
            c2 = m.GetConformers()[0]
            d2 = np.array(c2.GetPositions())
            binding_parts = []
            not_in_binding = [i for i in range(0, n2)]
            for bound_box in pockets:
                if len(constructed_graphs) >= 4:
                    break
                x_min = bound_box.x_range[0]
                x_max = bound_box.x_range[1]
                y_min = bound_box.y_range[0]
                y_max = bound_box.y_range[1]
                z_min = bound_box.z_range[0]
                z_max = bound_box.z_range[1]
                binding_parts_atoms = []
                idxs = []
                for idx, atom_cord in enumerate(d2):
                    if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                        binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                        idxs.append(idx)
                        if idx in not_in_binding:
                            not_in_binding.remove(idx)

                ami = am[np.array(idxs)[:, None], np.array(idxs)]
                g = nx.from_numpy_array(ami)

                for component in list(nx.connected_components(g)):
                    if len(component) <=10:
                        g.remove_nodes_from(component)

                remaining_node_indices = list(g.nodes)
                selected_atoms = [complete_molecule.GetAtomWithIdx(idx) for idx in remaining_node_indices]
                H = get_atom_feature(selected_atoms)

                graph = dgl.from_networkx(g)
                graph.ndata['h'] = torch.Tensor(H)
                graph = dgl.add_self_loop(graph)
                graph.ndata['pocket_number'] = torch.full((graph.number_of_nodes(),), 1, dtype=torch.float)
                constructed_graphs.append(graph)
                binding_parts.append(binding_parts_atoms)

            
        while len(constructed_graphs) < max_subpockets:
            virtual_graph = dgl.graph(([], []))
            virtual_graph.add_nodes(1)
            virtual_graph.ndata['h'] = torch.zeros((1, 31))
            virtual_graph.ndata['pocket_number'] = torch.full((virtual_graph.number_of_nodes(),), -1, dtype=torch.float)
            constructed_graphs.append(virtual_graph)


        constructed_graphs = dgl.batch(constructed_graphs)

        return constructed_graphs
            