import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from datasets.SemanticKITTI_dataloader.SemanticKITTITemporal import TemporalKITTISet
from utils.collations import SparseSegmentCollation
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']

class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.args.SemanticKITTI_path,
            seqs=[ '00', '01', '02', '03', '04', '05', '06', '07', '09', '10' ],
            split='train',
            resolution=0.05,
            num_points=180000,
            max_range=50.0,
            dataset_norm=False,
            std_axis_norm=False)
        loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True,
                            num_workers=4, collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.args.SemanticKITTI_path,
            seqs=[ '08' ],
            split='validation',
            resolution=0.05,
            num_points=180000,
            max_range=50.0,
            dataset_norm=False,
            std_axis_norm=False)
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            shuffle=False,
                            num_workers=4, 
                            collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.args.SemanticKITTI_path,
            seqs=[ '08' ],
            split='validation',
            resolution=0.05,
            num_points=180000,
            max_range=50.0,
            dataset_norm=False,
            std_axis_norm=False)
        loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=collate)
        return loader

dataloaders = {
    'KITTI': TemporalKittiDataModule,
}

