from data_provider.data_loader import FinanceHorizontalIteration, FinanceVerticalIteration, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
import numpy as np
from torch.utils.data import DataLoader
import torch

data_dict = {
    'Futures': FinanceHorizontalIteration,
    'FinanceVertical': FinanceVerticalIteration,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


class PadCollate:
    """
    Completing the batch to a fixed size by padding with zeros.
    Returning a vector mask to indicate which entries are real (1) and which
    are padding (0).
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, batch):
        actual_batch_size = len(batch)

        # unpack the batch into separate lists
        batch_x, batch_y, batch_x_mark, batch_y_mark = zip(*batch)

        # Convert to tensors if they are NumPy arrays
        batch_x = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in batch_x]
        batch_y = [torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y for y in batch_y]
        batch_x_mark = [torch.tensor(xm, dtype=torch.float32) if isinstance(xm, np.ndarray) else xm for xm in
                        batch_x_mark]
        batch_y_mark = [torch.tensor(ym, dtype=torch.float32) if isinstance(ym, np.ndarray) else ym for ym in
                        batch_y_mark]

        # Stack
        batch_x = torch.stack(batch_x)
        batch_y = torch.stack(batch_y)
        batch_x_mark = torch.stack(batch_x_mark)
        batch_y_mark = torch.stack(batch_y_mark)

        # Padding logic
        pad_len = self.batch_size - actual_batch_size
        if pad_len > 0:
            shape_x = list(batch_x.shape)
            shape_y = list(batch_y.shape)
            shape_x_mark = list(batch_x_mark.shape)
            shape_y_mark = list(batch_y_mark.shape)

            shape_x[0] = shape_y[0] = shape_x_mark[0] = shape_y_mark[0] = pad_len

            pad_x = torch.zeros(shape_x, dtype=batch_x.dtype)
            pad_y = torch.zeros(shape_y, dtype=batch_y.dtype)
            pad_x_mark = torch.zeros(shape_x_mark, dtype=batch_x_mark.dtype)
            pad_y_mark = torch.zeros(shape_y_mark, dtype=batch_y_mark.dtype)

            batch_x = torch.cat([batch_x, pad_x], dim=0)
            batch_y = torch.cat([batch_y, pad_y], dim=0)
            batch_x_mark = torch.cat([batch_x_mark, pad_x_mark], dim=0)
            batch_y_mark = torch.cat([batch_y_mark, pad_y_mark], dim=0)

        # Mask to indicate which entries are real (1) and which are padding (0)
        mask = torch.tensor([1] * actual_batch_size + [0] * pad_len, dtype=torch.uint8)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, mask


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False # False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        collate_fn = PadCollate(batch_size)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
