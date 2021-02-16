import torch
import os
import json
import shutil
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub
from torch_geometric.io import read_tu_data
from ogb.graphproppred import PygGraphPropPredDataset


class ZipDataset(torch.utils.data.Dataset):
    """
    This Dataset takes n datasets and "zips" them. When asked for the i-th element, it returns the i-th element of
    all n datasets. The lenght of all datasets must be the same.
    """
    def __init__(self, *datasets):
        """
        Stores all datasets in an internal variable.
        :param datasets: An iterable with PyTorch Datasets
        """
        self.datasets = datasets

        assert len(set(len(d) for d in self.datasets)) == 1

    def __getitem__(self, index):
        """
        Returns the i-th element of all datasets
        :param index: the index of the data point to retrieve
        :return: a list containing one element for each dataset in the ZipDataset
        """
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])


class ConcatFromListDataset(InMemoryDataset):
    """Create a dataset from a `torch_geometric.Data` list.
    Args:
        data_list (list): List of graphs.
    """
    def __init__(self, data_list):
        super(ConcatFromListDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class DatasetInterface:

    name = None

    @property
    def dim_node_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):

    # Do not implement a dummy init function that calls super().__init__, ow it breaks

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class KarateClubDatasetInterface(KarateClub, DatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__()
        self.root = root
        self.name = name
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.data.x = torch.ones(self.data.x.shape[0], 1)
        torch.save((self.data, self.slices), osp.join(self.processed_dir, 'data.pt'))

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def dim_node_features(self):
        return 1

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 2

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class LinkPredictionKarateClub(KarateClubDatasetInterface):

    @property
    def dim_target(self):
        return 1

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class PlanetoidDatasetInterface(Planetoid, DatasetInterface):

    # Do not implement a dummy init function that calls super().__init__, ow it breaks

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class LinkPredictionPlanetoid(PlanetoidDatasetInterface):

    @property
    def dim_target(self):
        return 1

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()



class ModelNet(InMemoryDataset, DatasetInterface):
    urls = {
        'ModelNet10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        'ModelNet40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    # Face to edge is the default to return a graph. Do not change it!
    def __init__(self, root, name='ModelNet10', train=True, transform=None,
                 pre_transform=FaceToEdge(), pre_filter=None):
        assert name in ['ModelNet10', 'ModelNet40']
        self.name = name
        super(ModelNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        if not os.path.exists(self.root, f'{self.name}.zip'):
            path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        # modified wrt original code to comply with full names
        folder = osp.join(self.root, self.name)
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        # Modify original code to merge train and test (for compatibility with library)
        x, edge_index, edge_slice, pos, y = None, None, None, None, None
        data_list = []
        for set in ['train', 'test']:
            data_list.extend(self.process_set(set))
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])



    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.x = data.pos
                setattr(data, 'pos', None)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        data_list = [self.pre_transform(d) for d in data_list]

        return data_list

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))


class MNISTSuperpixelsMoNet(InMemoryDataset, DatasetInterface):
    url = ('https://ls7-www.cs.tu-dortmund.de/fileadmin/ls7-www/misc/cvpr/mnist_superpixels.tar.gz')

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def dim_node_features(self):
        return 3

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 10

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

    # Put everything in a single data.pt file.
    # If you want to recover the original train/test split, simply split without shuffling
    def process(self):
        x, edge_index, edge_slice, pos, y = None, None, None, None, None
        for raw_path in self.raw_paths:
            if x is None:
                x, edge_index, edge_slice, pos, y = torch.load(raw_path)
                print(x.shape)
                print(edge_index.shape)
                print(edge_slice.shape)
                print(pos.shape)
                print(y.shape)
                print('-----')
            else:
                x2, edge_index2, edge_slice2, pos2, y2 = torch.load(raw_path)
                x = torch.cat((x, x2), dim=0)
                edge_index = torch.cat((edge_index, edge_index2), dim=1)
                edge_slice2 = edge_slice2[1:]  # remove first value 0
                edge_slice = torch.cat((edge_slice, edge_slice2+edge_slice[-1]), dim=0)
                pos = torch.cat((pos, pos2), dim=0)
                y = torch.cat((y, y2), dim=0)

                print(x.shape)
                print(edge_index.shape)
                print(edge_slice.shape)
                print(pos.shape)
                print(y.shape)

        edge_index, y = edge_index.to(torch.long), y.to(torch.long)
        m, n = y.size(0), 75
        x, pos = x.view(m * n, 1), pos.view(m * n, 2)
        node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
        graph_slice = torch.arange(m + 1, dtype=torch.long)
        self.data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice,
            'pos': node_slice
        }

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [d for d in data_list if self.pre_filter(d)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])


class PermutedMNISTSuperpixelsMoNet(MNISTSuperpixelsMoNet, DatasetInterface):
    """
    Extends MNISTSuperpixelsMoNet to generate permutations for each graph.
    IMPORTANT: we assume the number of nodes is the same for all graphs
    """

    def __init__(self, root, name, n_tasks, transform=None, pre_transform=None, pre_filter=None):
        self.n_tasks = n_tasks
        assert n_tasks > 0
        super(PermutedMNISTSuperpixelsMoNet, self).__init__(root, name, transform, pre_transform, pre_filter)

        self.permutation_to_apply = None
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.permutations_list = torch.load(self.processed_paths[1])
        assert n_tasks <= len(self.permutations_list), f"The dataset was generated with less ({len(self.permutations_list)}) permutations!"

    @property
    def processed_file_names(self):
        return ['data.pt', 'permutations.pt']

    def download(self):
        super().download()

    def process(self):
        super().process()
        no_nodes = 75  # hardcoded for this dataset
        print(f"Generating {self.n_tasks} permutations...")
        permutations_list = [torch.randperm(no_nodes) for _ in range(self.n_tasks)]
        torch.save(permutations_list, self.processed_paths[1])

    def apply_permutation(self, task_id):
        self.permutation_to_apply = task_id

    def get(self, idx):
        data = super().get(idx)
        if hasattr(self, "permutations_list") and self.permutation_to_apply is not None:
            # I have already created the dataset
            perm = self.permutations_list[self.permutation_to_apply]
            # This will NOT permute the original data.
            data.x = data.x[perm, :]
            data.pos = data.pos[perm, :]

        return data


class CLSplitDataset:

    def get_indices_of_targets(self, class_list):
        raise NotImplementedError("You should subclass CLSplitDataset and implement this method")


class GNNBenchmarkDataset(InMemoryDataset, DatasetInterface, CLSplitDataset):
    names = ['MNIST', 'CIFAR10']
    url = 'https://pytorch-geometric.com/datasets/benchmarking-gnns'

    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, use_node_attr=False,
                 use_edge_attr=False, cleaned=False):
        self.name = name
        assert self.name in self.names

        super(GNNBenchmarkDataset, self).__init__(root, transform,
                                                  pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def dim_node_features(self):
        return self.data.x.shape[1] + self.data.pos.shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 10

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name
        return [f'{name}_train.pt', f'{name}_val.pt', f'{name}_test.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_indices_of_targets(self, class_list):
        targets = self.data.y
        indices = torch.nonzero(torch.sum(torch.stack([targets == c for c in class_list], dim=0), dim=0), as_tuple=False).squeeze().int().tolist()
        return indices

    def download(self):
        url = f'{self.url}/{self.name}.zip'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = []
        for i in range(3):
            self.data, self.slices = torch.load(self.raw_paths[i])
            data_list.extend([self.get(i) for i in range(len(self))])

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))



class OGBG(PygGraphPropPredDataset, DatasetInterface, CLSplitDataset):
    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, meta_dict=None):
        super().__init__('-'.join(name.split('_')), root, transform, pre_transform, meta_dict)  #
        self.name = name
        self.data.y = self.data.y.squeeze()

    @property
    def dim_node_features(self):
        return 1

    @property
    def dim_edge_features(self):
        if self.data.edge_attr is not None:
            return self.data.edge_attr.shape[1]
        else:
            return 0

    @property
    def dim_target(self):
        return 37

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_indices_of_targets(self, class_list):
        targets = self.data.y
        indices = torch.sum(torch.stack([targets == c for c in class_list], dim=0), dim=0).nonzero().squeeze().int().tolist()
        return indices

    def download(self):
        from ogb.utils.url import decide_download, download_url, extract_zip
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            print(f'Removing {path}')
            os.unlink(path)
            print(f'Removing {self.root}')
            shutil.rmtree(self.root)
            print(f'Moving {osp.join(self.original_root, self.download_name)} to {self.root}')
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

    def process(self):
        super().process()
