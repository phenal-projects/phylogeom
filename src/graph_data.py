import glob
import os.path as osp
import pickle
import re
import subprocess as sp
import uuid
from collections import Counter
from functools import reduce
from itertools import combinations

import dendropy
import dendropy as d
import h5py
import numpy as np
import torch
import torch_geometric.utils as utils
from Bio import AlignIO
from Bio import Phylo
from Bio import SeqIO
from Bio.Phylo import TreeConstruction
from Bio.Phylo.Applications import RaxmlCommandline
from scipy import sparse, optimize
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

temp_data = "../data/temp/"

res_to_index = {
    "-": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "X": 20,
    "Y": 21,
}

# embeddings for aminoacids
aesnn3 = np.array(
    [[0.0, 0.0, 0.0],  # -
     [-.99, -.61, 0.00],  # A
     [0.34, 0.88, 0.35],  # C
     [0.74, -.72, -.35],  # D
     [0.59, -.55, -.99],  # E
     [0.87, 0.65, -.53],  # F
     [-.79, -.99, 0.10],  # G
     [0.08, -.71, 0.68],  # H
     [-.77, 0.67, -.37],  # I
     [-.63, 0.25, 0.50],  # K
     [-.92, 0.31, -.99],  # L
     [-.80, 0.44, -.71],  # M
     [0.77, -.24, 0.59],  # N
     [-.99, -.99, -.99],  # P
     [0.12, -.99, -.99],  # Q
     [0.28, -.99, -.22],  # R
     [0.99, 0.40, 0.37],  # S
     [0.42, 0.21, 0.97],  # T
     [-.99, 0.27, -.52],  # V
     [-.13, 0.77, -.90],  # W
     [-.06, -.08, -.24],  # X
     [0.59, 0.33, -.99]]  # Y
)

reconstruction_methods = {
    "upgma": 0,
    "nj": 1,
    "raxml": 2
}


class Multipartition:
    def __init__(self, clade, taxons):
        """
        Multipartition representation for given clade
        :param clade: a clade to represent
        :param taxons: set of all taxons in the tree
        """
        self.parts = list()
        self.taxons = taxons
        for child in clade:
            self.parts.append(set((term.name for term in child.get_terminals())))
        self.parts.append(taxons - reduce(lambda a, b: a.union(b), self.parts))

    def dist(self, other):
        """Computes distance between two multipartitions. Complexity -- O(max(N, M)^3)"""
        assert (type(other) == type(self))
        if self.taxons == other.taxons:
            print("Warning! Sets of leafs are not equal")
            print(self.taxons)
            print(other.taxons)
        mat_size = max(len(self.parts), len(other.parts))
        sim_mat = np.zeros((mat_size, mat_size), dtype=np.long)  # dissim mat
        for ia, a in enumerate(self.parts):
            sim_mat[ia] = len(a)
            for ib, b in enumerate(other.parts):
                sim_mat[ia, ib] = len(a.symmetric_difference(b))
        row_ind, col_ind = optimize.linear_sum_assignment(sim_mat)
        return sim_mat[row_ind, col_ind].sum()

    def __len__(self):
        return len(self.parts)


def multipartitions(tree):
    """
    Multipartitions representations of clades in the tree
    :param tree: tree, for which representations will be computed
    :return: MP representations of the inner clades
    """
    representations = list()
    taxons = set(tree.get_terminals())
    for clade in tree.get_nonterminals():
        clade_repr = Multipartition(clade, taxons)
        representations.append(clade_repr)
    return representations


def get_node_representation(multipart, seq_dict):
    res = torch.zeros(231)  # pooling max divergence
    if seq_dict is None:
        return res
    freqs = list()
    for st in multipart.parts:
        freqs.append(Counter())
        for sq in st:
            if sq in seq_dict.keys():
                freqs[-1] += Counter(seq_dict[sq])
            else:
                print('Skip absent {}'.format(sq))
    res_pos = 0
    for aapair in combinations(res_to_index.keys(), 2):
        for grpair in combinations(range(len(multipart)), 2):
            na = sum(freqs[grpair[0]].values())
            nb = sum(freqs[grpair[1]].values())
            # print(na, nb)
            if na * nb != 0:
                sr = abs(float(freqs[grpair[0]][aapair[0]]) / na - float(freqs[grpair[1]][aapair[1]]) / nb)
                if res[res_pos] < sr:
                    res[res_pos] = sr
        res_pos += 1
    return res


def to_coo(tree, target_tree, seq_dict):
    """
    Transforms tree to graph coo format
    :param tree: Phylo tree to convert
    :param target_tree: Phylo tree with target tree (to evaluate clades)
    :param seq_dict: dict (name -> sequence for feature construction)
    :return: COO matrix, normalized edge lengths, clades' quality, lookup dict (Name to id)
    """
    # all target bit strings
    target_mps = multipartitions(target_tree)
    clades = list(tree.find_clades(order='level'))
    term_names = [term.name for term in tree.get_terminals()]
    term_names.sort()
    # lookup table (node name to index)
    lookup = dict()
    for cid, clade in enumerate(clades):
        if clade.name is None:
            clade.name = str(cid)
        lookup[clade.name] = cid
    # init data
    x = torch.ones((len(clades), 231))
    coo = torch.zeros((2, len(clades) - 1), dtype=torch.long)
    y = torch.zeros((len(clades), 1), dtype=torch.float)
    lengths = torch.ones((len(clades) - 1, 1))
    cid = 0
    taxons = set((q.name for q in tree.get_terminals()))
    for num_clade, parent in enumerate(tree.get_nonterminals()):
        mp = Multipartition(parent, taxons)
        x[num_clade] = get_node_representation(mp, seq_dict)
        # distance to the most similar branch
        y[lookup[parent.name]] = float(min([mp.dist(tmp) for tmp in target_mps]))
        for child in parent.clades:
            coo[0, cid] = lookup[parent.name]
            coo[1, cid] = lookup[child.name]
            if child.branch_length:
                lengths[cid] = child.branch_length
            cid += 1

    if not tree.rooted:
        coo = torch.cat((coo, coo[[1, 0]]), dim=1)
        lengths = torch.cat((lengths, lengths), dim=0)
    return x, coo, lengths / torch.max(lengths), torch.pow(0.5, y), lookup


def fastml_asr(tree_file, aln_file, prefix='./'):
    """
    Generates ancestral sequences with fastml and given tree
    :param tree_file: newick tree file
    :param aln_file: fasta file with the MSA
    :param prefix: directory to save the results
    :return: fasta file, newick file with tree with inner nodes
    """
    print(tree_file)
    print(aln_file)
    t = d.Tree.get(path=tree_file, schema='newick')
    uniq_name = str(uuid.uuid4())
    output_name = osp.join(prefix, uniq_name + '.fasta')
    output_tree = osp.join(prefix, uniq_name + '.tre')
    t.write_to_path(osp.join(prefix, 'temp_tree.tre'), 'newick', suppress_internal_node_labels=True)
    r = sp.run([
        'fastml', '-qf', '-ma', '-b',
        '-s', aln_file,
        '-t', osp.join(prefix, 'temp_tree.tre'),
        '-j', output_name,
        '-x', output_tree
    ])
    if r.returncode != 0:
        print(r.returncode)
        print("############STDOUT###########")
        print(r.stdout)
        print("############STDERR###########")
        print(r.stderr)
        raise ChildProcessError('Something went wrong in fastml process')
    return output_name, output_tree


class Trees(InMemoryDataset):
    def __init__(self, root, alns_dir, target_tree, transform=None, pre_transform=None):
        """
        Dataset with a lot of trees. Everybody likes trees!
        :param root: path to working directory, from where data will be loaded and where data will be stored
        :param target_tree: target tree. The ultimate truth
        :param transform: transforms to apply but not to save
        :param pre_transform: transforms to apply and save
        """
        self.target_tree = target_tree
        self.alns_dir = alns_dir
        self.seqs = None
        super(Trees, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[1], 'rb') as fin:
            self.seqs = pickle.load(fin)

    @property
    def raw_file_names(self):
        return glob.glob(osp.join(self.root, "*.tre"))

    @property
    def processed_file_names(self):
        return ['data.pt', 'seq_links.pickle']

    def download(self):
        pass

    def process(self):
        """
        Loads files from the disk and processes them
        :return: None
        """
        data_list = list()
        self.seqs = list()
        for file in self.raw_file_names:
            m = re.search(r'(\d{3}_)?PF\d{5}_\d+', file)
            if m is None:
                continue  # skip files with no sequences
            aln = osp.join(self.alns_dir, m.group(0) + '.fasta')
            # print(aln)
            # alternative features. Must be eliminated soon
            # sequences, new_tree_path = fastml_asr(file, aln, osp.join(self.root, 'fml_output'))
            tree = Phylo.read(file, 'newick')
            with open(aln) as fin:
                seqs = SeqIO.parse(fin, 'fasta')
                seq_dict = {seq.name: seq.seq for seq in seqs}
            features, ei, el, y, lud = to_coo(tree, self.target_tree, seq_dict)
            self.seqs.append((aln, lud))  # stores sequences of the corresponding Data objects

            data_list.append(Data(x=features, edge_index=ei, edge_attr=el, y=y))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)  # to internal format
        torch.save((self.data, self.slices), self.processed_paths[0])
        with open(self.processed_paths[1], 'wb') as fout:
            pickle.dump(self.seqs, fout)


# noinspection PyUnresolvedReferences
def load_ready_trees(root, pattern, target_tree, transform=None):
    """
    Please, do not abuse this method! It may cause terrible and unpredictable errors
    Used once, it should be replaced with the constructor.
    :param root: root directory of the dataset
    :param pattern: pattern (starred-expression) of tree files. Only .tre and .nwk extensions are supported
    :param target_tree: target tree. The ultimate truth
    :param transform: transforms to apply but not to save
    :return: Trees dataset
    """

    class Empty:
        pass

    res = Empty()
    res.__class__ = Trees  # bypassing int and processing
    data_list = list()
    res.seqs = list()
    res.root = root
    res.target_tree = target_tree
    res.alns_dir = root
    res.transform = transform
    res.processed_dir = osp.join(res.root, 'processed')
    for file in glob.glob(pattern):
        tree = Phylo.read(file, 'newick')
        _, ei, el, y, lud = to_coo(tree, target_tree, None)
        data_list.append(Data(x=torch.ones((torch.max(ei).item() + 1, 1)), edge_index=ei, edge_attr=el, y=y))
        res.seqs.append((file[:-3] + 'fasta', lud))  # stores sequences of the corresponding Data objects
    for data, (seq_path, lookup) in zip(data_list, res.seqs):
        feats = np.load(seq_path[:-5] + 'npy')
        with open(seq_path) as f:
            taxon_order = list(map(lambda x: lookup[x[1:].strip()], filter(lambda x: x[0] == '>', f)))
            data.x = torch.from_numpy(feats[np.argsort(taxon_order)])
    res.data, res.slices = res.collate(data_list)  # to internal format
    torch.save((res.data, res.slices), res.processed_paths[0])
    with open(res.processed_paths[1], 'wb') as fout:
        pickle.dump(res.seqs, fout)
    return res


class SeqDataset(Dataset):
    def __init__(self, fastas: list or None, max_length: int or None, save_path: str or None = './seq_data.pt'):
        """
        Dataset with gapped sequences. You may use it to train your sequence embedding model
        :param fastas: filenames of input files
        :param max_length: maximum length of an alignment
        :param save_path: file, where processed data will be saved
        """
        if fastas is None:
            self.data = None
        else:
            self.data = []
            for i, fasta in enumerate(fastas):
                aln = AlignIO.read(fasta, "fasta")
                for row_num, row in enumerate(aln):
                    seq = torch.zeros(max_length, dtype=torch.long)
                    for col_num, res in enumerate(row):
                        seq[col_num] = res_to_index[res]
                    self.data.append(seq)
            self.data = torch.stack(self.data)
            if not (save_path is None):
                with open(save_path, 'wb') as f:
                    torch.save(self.data, f)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def save(self, path):
        """
        Saves dataset to a disk
        :param path: path to .pt file
        :return: None
        """
        with open(path, 'wb') as f:
            torch.save(self.data, f)

    @staticmethod
    def load(path):
        """
        Loads dataset into the memory
        :param path: path to .pt file
        :return: a SeqDataset object
        """
        # noinspection PyTypeChecker
        r = SeqDataset(None, None, None)
        r.data = torch.load(path)
        return r


class KNNbyAttr:
    def __init__(self, k=6, force_undirected=True):
        """
        Returns undirected KNN graph
        :param k: K - numbers of edges of every node to keep
        :param force_undirected: if true, results will be undirected graph
        """
        self.k = k
        self.force_undirected = force_undirected

    def __call__(self, data):
        """
        Computes the transform on the data
        !needs optimization (is it even correct?)!
        :param data: given data
        :return: new Data object
        """
        new_index = []
        new_attr = []
        for node in range(torch.max(data.edge_index).item() + 1):
            candidates = data.edge_index[1][data.edge_index[0] == node]
            kstat = -np.partition(-data.edge_attr[data.edge_index[0] == node].numpy(), self.k)[self.k]
            [
                (
                    new_index.append([node, i]),
                    new_attr.append(data.edge_attr[(data.edge_index[0] == node) * (data.edge_index[1] == i)][0]),
                ) for i in candidates if
                data.edge_attr[(data.edge_index[0] == node) * (data.edge_index[1] == i)] <= kstat
            ]
        data.edge_index = torch.tensor(new_index).t().contiguous()
        data.edge_attr = torch.tensor(new_attr)
        if self.force_undirected:
            new_index = utils.to_undirected(data.edge_index)
            new_attr = [
                data.edge_attr[(data.edge_index[0] == i) * (data.edge_index[1] == j)] for i, j in zip(
                    new_index[0], new_index[1]
                )
            ]
        data.edge_index = torch.tensor(new_index).t().contiguous()
        data.edge_attr = torch.tensor(new_attr)
        return data


class Aln(Dataset):
    def __init__(self, fastas, labels, max_seqs=45):
        """
        Creates dataset object from given alignments. Drops ones with len>max_lens.
        You can create an empty dataset to load your data later. It is more memory-friendly approach.
        :param fastas: alignment files
        :param labels: labels of each fasta
        :param max_seqs: maximum number of sequences in one file
        """
        self.is_sparse = False
        self.edge_indices = None
        self.edge_weights = None
        self.f = None  # holder for hdf5 file
        self.data = np.zeros((len(fastas), max_seqs, 3))
        self.labels = np.array(labels, dtype=np.long)
        self.dist = np.zeros((len(fastas), max_seqs, max_seqs))
        calc = TreeConstruction.DistanceCalculator('blosum62')  # calculator for distance matrices
        for file_index, file in enumerate(fastas):
            aln = AlignIO.read(file, "fasta")
            for seq_index, seq in enumerate(aln):
                for col_index, col in enumerate(seq):
                    self.data[file_index, seq_index] += aesnn3[res_to_index[col]]
            self.dist[file_index, :len(aln), :len(aln)] = np.array(calc.get_distance(aln))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            return self.data[idx], self.edge_indices[idx], self.edge_weights[idx], self.labels[idx]
        return self.data[idx], self.dist[idx], self.labels[idx]

    def save(self, path):
        """Saves the data to specified hdf5 file"""
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "alns",
                data=self.data
            )
            f.create_dataset(
                "labels",
                data=self.labels
            )
            f.create_dataset(
                "dist",
                data=self.dist
            )

    @staticmethod
    def load_lazy(path):
        """Constructs dataset as a wrapper for stored data"""
        res = Aln([], [])
        res.f = h5py.File(path, "r")
        res.data = res.f["alns"]
        res.labels = res.f["labels"]
        res.dist = res.f["dist"]
        return res

    def nolazy(self):
        """Loads data into RAM completely"""
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.dist = np.array(self.dist)

    def close(self):
        self.f.close()

    def sparse(self):
        """Returns sparse representations of stored graphs"""
        self.is_sparse = True
        self.edge_indices = np.stack(list(map(
            lambda x: np.stack(sparse.csr_matrix(x).nonzero()),
            self.dist + 1  # no zeros allowed to avoid shape mismatch
        )))  # ineffective, but acceptable
        self.edge_weights = np.stack(list(map(
            lambda x: sparse.csr_matrix(x).data,
            self.dist + 1
        ))) - 1


class AlignDataset(InMemoryDataset):
    def __init__(self, aln, transform=None, pre_transform=None):
        """
        Dataset with complete graph with distance matrix from proteins
        :param aln: Aln dataset
        :param transform: a transform to compute
        :param pre_transform: a transform to save to hte HDD
        """
        super(AlignDataset, self).__init__("./", transform, pre_transform)
        if not isinstance(aln, Aln):
            raise TypeError("Aln dataset was expected")
        self.aln = aln
        self.data = None
        self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        if not self.aln.is_sparse:
            self.aln.sparse()
        self.data = [Data(
            x=torch.tensor(self.aln.data[i], dtype=torch.float),
            edge_index=torch.tensor(self.aln.edge_indices[i], dtype=torch.long),
            edge_attr=torch.tensor(self.aln.edge_weights[i], dtype=torch.float).reshape(-1, ),
            y=torch.tensor(self.aln.labels[i], dtype=torch.long).reshape(1, )
        ) for i in range(len(self.aln))]

        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def labeler(files, etalon_tree, tree_path=".", rebuild=False):
    """
    Constructs labels for given files. (Best phylogeny reconstruction method)
    :param files: an iterable with file paths to alignments
    :param etalon_tree: the path to etalon tree
    :param tree_path: a directory, where built trees will be stored
    :param rebuild: set it True, if you need to rebuild trees or build them from scratch
    :return: tensor with labels
    """
    if rebuild:
        calculator = TreeConstruction.DistanceCalculator('blosum62')
        dist_constructor = TreeConstruction.DistanceTreeConstructor()

        # construct all trees with UPGMA, NJ and raxml
        for i, file in enumerate(files):
            aln = AlignIO.read(file, 'fasta')
            tree = dist_constructor.upgma(calculator.get_distance(aln))
            name = file.split("/")[-1].split(".")[0]
            Phylo.write(
                tree,
                osp.join(tree_path, 'upgma_{}.tre'.format(name)),
                'newick'
            )
            tree = dist_constructor.nj(calculator.get_distance(aln))
            Phylo.write(
                tree,
                osp.join(tree_path, 'nj_{}.tre'.format(name)),
                'newick'
            )
            raxml = RaxmlCommandline(
                sequences=file,
                model='PROTCATWAG',
                name='{}.tre'.format(name),
                threads=3,
                working_dir=tree_path
            )
            _, stderr = raxml()
            print(stderr)
            print('{} finished'.format(name))
    # get best tree
    tns = dendropy.TaxonNamespace()
    act_tree = dendropy.Tree.get_from_path(
        osp.join(tree_path, etalon_tree), "newick", taxon_namespace=tns
    )
    act_tree.encode_bipartitions()
    distances = np.zeros(shape=(len(files), 3))
    for i, file in enumerate(files):
        name = file.split("/")[-1].split(".")[0]
        nj_tree = dendropy.Tree.get_from_path(
            osp.join(tree_path, "nj_{}.tre".format(name)), "newick", taxon_namespace=tns
        )
        up_tree = dendropy.Tree.get_from_path(
            osp.join(tree_path, "upgma_{}.tre".format(name)), "newick", taxon_namespace=tns
        )
        ml_tree = dendropy.Tree.get_from_path(
            osp.join(tree_path, "RAxML_bestTree.{}.tre".format(name)), "newick", taxon_namespace=tns
        )
        distances[i, 0] = dendropy.calculate.treecompare.symmetric_difference(
            nj_tree, act_tree
        )
        distances[i, 1] = dendropy.calculate.treecompare.symmetric_difference(
            up_tree, act_tree
        )
        distances[i, 2] = dendropy.calculate.treecompare.symmetric_difference(
            ml_tree, act_tree
        )
    return distances.argmin(1)


def balanced_classes_weights(dataset, nclasses):
    """
    Computes weights for balanced sampling
    :param dataset: dataset (Geometric Pytorch InMemoryDataset or Dataset and their subclasses)
    :param nclasses: total number of classes in dataset
    :return: list of weights
    """
    count = [0] * nclasses
    for item in dataset:
        count[item.y] += 1
    weight_per_class = [0.] * nclasses
    length = float(len(dataset))
    for i in range(nclasses):
        weight_per_class[i] = length / float(count[i])
    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val.y]
    return weight
