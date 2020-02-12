# create fml_output dir first. NEEDS FIX!
from Bio import Phylo as phy

from src import graph_data as gd

data_path = '/home/ilbumi/PycharmProjects/geometric_phylogeny/data/'
target_tree = phy.read(data_path + "tree/arch_fung/Archaea.tre", 'newick')
trees = gd.Trees(data_path + "tree/Archaea/", data_path + "alns/Archaea/", target_tree)
