import torch
import dendropy
from Bio import SeqIO, AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from dendropy.calculate.treecompare import symmetric_difference
from torch_geometric.data import Data
import subprocess
from collections import defaultdict
from itertools import combinations


# methods list
phylip_symb = ['fproml', 'fpromlk']


def build_trees(alns, trees):
    """
    Builds trees from the alignment and stores them as trees
    :param alns: the list of alignments in fasta format
    :param trees: the path of built trees. Must contain one {} group to format method name
    :return: None
    """
    # prepare calculator and constructor
    calculator = DistanceCalculator('blosum62')
    constructor = DistanceTreeConstructor()
    for aln, tree in zip(alns, trees):
        print(aln, tree)
        processes = []
        for method in phylip_symb:
            processes.append(subprocess.Popen([
                method,
                '-auto',
                '-sequence',
                aln,
                '-outtreefile',
                tree.format(method)
            ]))
        # nj + upgma
        with open(aln) as fin:
            alnr = AlignIO.read(fin, 'fasta')
        dm = calculator.get_distance(alnr)
        Phylo.write(
            constructor.upgma(dm),
            tree.format('upgma'),
            'newick'
        )
        Phylo.write(
            constructor.nj(dm),
            tree.format('nj'),
            'newick'
        )
        for process in processes:
            print(process.wait())


class PairData(Data):
    def __init__(self, edge_ind1, edge_ind2, x1, x2, y, lookup1, lookup2, aln):
        super().__init__()
        self.edge_ind1 = edge_ind1
        self.edge_ind2 = edge_ind2
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.lookup1 = lookup1
        self.lookup2 = lookup2
        self.aln = aln

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x1.size(0)
        if key == 'edge_index_t':
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


def process_trees(trees, etalon_tree, alignments, seq_to_tensor, tensor_length=64):
    data_dict = defaultdict(list)
    taxa = dendropy.TaxonNamespace()
    etalon_tree = dendropy.Tree.get(
        path=etalon_tree,
        schema='newick',
        taxon_namespace=taxa,
    )
    tree_yielder = dendropy.Tree.yield_from_files(
        files=trees,
        schema='newick',
        taxon_namespace=taxa,
    )
    for tree, aln_file in zip(tree_yielder, alignments):
        print(tree, aln_file)
        # reading tree
        # reading alignment
        with open(aln_file) as fin:
            seq_dict = {seq.name: seq.seq for seq in SeqIO.parse(fin, 'fasta')}
        # create tensors
        nlen = len(tree.nodes())
        x = torch.zeros(nlen, tensor_length)
        # lookup table for TAXON->INDEX mapping
        lookup = {str(t): tid for tid, t in enumerate(tree.preorder_node_iter())}
        coo = torch.zeros((2, len(tree.edges())), dtype=torch.long)
        edge_len = torch.zeros(len(tree.edges()), dtype=torch.long)
        edge_ind = 0
        correct_tree = True
        # iterating through every node to calculate representations and create coo
        for node in tree.preorder_node_iter():
            # add edges to immediate children
            for child_edge in node.child_edge_iter():
                coo[0, edge_ind] = lookup[str(child_edge.head_node)]
                coo[1, edge_ind] = lookup[str(child_edge.tail_node)]
                edge_len[edge_ind] = 1.0 if child_edge.length is None else child_edge.length
                edge_ind += 1
            # calculate features
            if node.taxon is None:
                x[lookup[str(node)]] = seq_to_tensor(None)
            else:
                try:
                    x[lookup[str(node)]] = seq_to_tensor(seq_dict[node.taxon.label])
                except KeyError as e:
                    print("BROKEN TREE/FILE: {}".format(aln_file))
                    print(e)
                    correct_tree = False
                    break
        if correct_tree:  # DO NOT BUILD WRONG TREES
            data_dict[aln_file].append(
                {
                    'x': x,
                    'edge_index': coo,
                    'edge_attr': edge_len,
                    'y': symmetric_difference(
                        etalon_tree.extract_tree_with_taxa_labels([r.taxon.label for r in tree.leaf_nodes()]), tree
                    ),
                    'lookup': lookup
                }
            )
    data_list = list()
    for aln in data_dict:
        for i, j in combinations(range(len(data_dict[aln])), 2):
            data_list.append(PairData(
                data_dict[aln][i]['edge_index'],
                data_dict[aln][j]['edge_index'],
                data_dict[aln][i]['x'],
                data_dict[aln][j]['x'],
                int(data_dict[aln][i]['y'] > data_dict[aln][j]['y']),
                data_dict[aln][i]['lookup'],
                data_dict[aln][j]['lookup'],
                aln
            ))

    return data_list
