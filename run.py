import argparse
import os
import time

import pandas as pd
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from finance_collection.constants import trd_days_year, trd_days_month
from utils.print_args import print_args
import random
import numpy as np
import sys


def main(args_in=None):
    random.seed(int(time.time()))
    np.random.seed(int(time.time()) % 2 ** 32)
    torch.manual_seed(int(time.time()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(time.time()))

    # Disable deterministic behavior
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # fix_seed = int(time.time()) # 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)
    # print(f"Seed used: {fix_seed}")

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')  # type=str, required=True
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train set ratio from entire date')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test set ratio from entire date')
    parser.add_argument('--limit_asset_number', type=int, default=0, help='limit the number of assets to be trained')
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_tru"
                                                            "e", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--start_date', type=str, default="",
                        help="Start Date of the training")  #########################
    parser.add_argument('--end_date', type=str, default="", help="End Date of the training")  #########################
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    parser.add_argument('--vol_target', type=float, default=0.15, help='Target volatility')

    args = parser.parse_args(args_in)

    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 60)
    pd.set_option('show_dimensions', True)
    pd.set_option('display.float_format', '{:16,.0f}'.format)

    """
    number_of_features_per_asset = 20 # 321
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args.task_name = 'long_term_forecast'
    args.num_workers=0
    args.train_epochs = 300
    args.is_training = 1
    args.root_path = './dataset/FinanceStrategiesFutures/'
    args.data_path=''   # electricity.csv
    args.model_id='Quandl_4Y_1Y_With_Shuffle'
    args.model='TemporalFusionTransformer' # DLinear
    args.data='FinanceVertical' # 'Futures'
    args.features='MS'
    args.target = 'target_returns'
    args.seq_len = trd_days_year#4 * trd_days_year  # 4 years
    args.label_len = trd_days_month # 1 * trd_days_year  # 1 year
    args.pred_len = trd_days_month # 1 * trd_days_year  # 1 year
    args.e_layers=2
    args.d_layers=1
    args.factor=3
    args.enc_in = number_of_features_per_asset
    args.dec_in = number_of_features_per_asset
    args.c_out = 1
    args.des='Exp'
    args.itr=1
    args.loss = 'Sharpe'  # Custom loss function for Sharpe ratio
    args.use_gpu = True if torch.cuda.is_available() else False
    args.start_date = '2014'
    args.end_date = '2020'
    """
    # main(sys.argv[1:])  # Automatically gets CLI args ############################################ MUST BE ON WHEN RUNNING VIA COMMAND LINE AND IN COLAB

    model_run_TFT_1 = ['--task_name=long_term_forecast', '--is_training=1', '--batch_size=256', # 64, 128, 256
                     '--model_id=TFT_epoch300_seq252_label21_pred1_workers10_dropout0.3_lr0.01', '--num_workers=0', '--seasonal_patterns=', '--patience=5',
                     '--root_path=./dataset/FinanceStrategiesFutures/', '--learning_rate=0.01',
                     '--data_path=', '--model=TemporalFusionTransformer',
                     '--data=FinanceVertical', '--features=MS', '--train_epochs=300',
                     '--target=target_returns', '--seq_len=252', '--label_len=21',
                     '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
                     '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
                     '--loss=Sharpe', '--start_date=2015', '--end_date=2020',
                       '--train_ratio=0.9', '--test_ratio=0.05', '--dropout=0.3',
                     '--limit_asset_number=4']
    #
    # model_run_TFT_2 = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_TFT_epoch300_label1_pred1_workers10', '--num_workers=10',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/',
    #                    '--data_path=', '--model=TemporalFusionTransformer',
    #                    '--data=FinanceVertical', '--features=MS', '--train_epochs=300',
    #                    '--target=target_returns', '--seq_len=252', '--label_len=1',
    #                    '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
    #                    '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
    #                    '--loss=Sharpe', '--start_date=2010', '--end_date=2020',
    #                    '--limit_asset_number=4']

    # model_run_TFT_3 = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_TFT_epoch300_label1_pred1_workers0_ma126', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/',
    #                    '--data_path=', '--model=TemporalFusionTransformer',
    #                    '--data=FinanceVertical', '--features=MS', '--train_epochs=300',
    #                    '--target=target_returns', '--moving_avg=126','--seq_len=252', '--label_len=1',
    #                    '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
    #                    '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
    #                    '--loss=Sharpe', '--start_date=2010', '--end_date=2015',
    #                    '--limit_asset_number=4']
    #
    # model_run_TFT_4 = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_TFT_epoch100_label1_pred1_train085_test005', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/',
    #                    '--data_path=', '--model=TemporalFusionTransformer', '--train_ratio=0.85', '--test_ratio=0.05',
    #                    '--data=FinanceVertical', '--features=MS', '--train_epochs=100',
    #                    '--target=target_returns', '--seasonal_patterns=Yearly', '--seq_len=252', '--label_len=1',
    #                    '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
    #                    '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
    #                    '--loss=Sharpe', '--start_date=2010', '--end_date=2015',
    #                    '--limit_asset_number=4']
    #
    # model_run_TFT_5 = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_TFT_epoch100_label1_pred1_train07_test015', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/', '--train_ratio=0.7', '--test_ratio=0.15',
    #                    '--data_path=', '--model=TemporalFusionTransformer',
    #                    '--data=FinanceVertical', '--features=MS', '--train_epochs=3000',
    #                    '--target=target_returns', '--seasonal_patterns=Yearly', '--seq_len=252', '--label_len=1',
    #                    '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
    #                    '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
    #                    '--loss=Sharpe', '--start_date=2010', '--end_date=2015',
    #                    '--limit_asset_number=4']
    #
    # model_run_TFT_6 = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_TFT_epoch3000_label1_pred1_dmodel1024', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/', '--d_model=1024',
    #                    '--data_path=', '--model=TemporalFusionTransformer',
    #                    '--data=FinanceVertical', '--features=MS', '--train_epochs=3000',
    #                    '--target=target_returns', '--seasonal_patterns=Yearly', '--seq_len=252', '--label_len=1',
    #                    '--pred_len=1', '--e_layers=2', '--d_layers=1', '--factor=3',
    #                    '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp', '--itr=1',
    #                    '--loss=Sharpe', '--start_date=2010', '--end_date=2015',
    #                    '--limit_asset_number=4']
    #
    # model_run_Patch = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_PatchTST_Improved', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/', '--train_epochs=300',
    #                    '--data_path=', '--model=PatchTST', '--data=FinanceVertical',
    #                    '--features=MS', '--target=target_returns', '--seq_len=252',
    #                    '--label_len=21', '--pred_len=21', '--e_layers=2', '--d_layers=1',
    #                    '--factor=3', '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp',
    #                    '--itr=1', '--loss=Sharpe', '--start_date=2014', '--end_date=2020']
    #
    # model_run_DLinear = ['--task_name=long_term_forecast', '--is_training=1',
    #                      '--model_id=Quandl_DLinear_Improved_0.7_train', '--num_workers=0',
    #                      '--root_path=./dataset/FinanceStrategiesFutures/', '--train_epochs=300',
    #                      '--data_path=', '--model=DLinear', '--data=FinanceVertical',
    #                      '--features=MS', '--target=target_returns', '--seq_len=252',
    #                      '--label_len=1', '--pred_len=1', '--e_layers=2', '--d_layers=1',
    #                      '--factor=3', '--enc_in=20', '--dec_in=20', '--c_out=1', '--des=Exp',
    #                      '--itr=1', '--loss=Sharpe', '--start_date=2014', '--end_date=2020']
    #
    # model_run_Mamba = ['--task_name=long_term_forecast', '--is_training=1',
    #                    '--model_id=Quandl_MambaSimple', '--num_workers=0',
    #                    '--root_path=./dataset/FinanceStrategiesFutures/', '--train_epochs=300',
    #                    '--data_path=', '--model=MambaSimple', '--data=FinanceVertical',
    #                    '--features=MS', '--target=target_returns', '--seq_len=252',
    #                    '--label_len=21', '--pred_len=21', '--e_layers=2', '--d_layers=1',
    #                    '--factor=3', '--enc_in=19', '--dec_in=20', '--c_out=1', '--des=Exp',
    #                    '--itr=1', '--loss=Sharpe', '--start_date=2014', '--end_date=2020']
    #
    # main(model_run_Patch)
    # main(model_run_DLinear)
    main(model_run_TFT_1)
    # main(model_run_TFT_2)
    # main(model_run_TFT_3)
    # main(model_run_TFT_4)
    # main(model_run_TFT_5)
    # main(model_run_TFT_6)
    # main(model_run_Mamba)

    # --task_name=long_term_forecast --is_training=1 --model_id=Quandl_TFT_Improved --num_workers=0 --root_path=./dataset/FinanceStrategiesFutures/ --data_path= --model=TemporalFusionTransformer --data=FinanceVertical --features=MS --train_epochs=300 --target=target_returns --seq_len=252 --label_len=1 --pred_len=1 --e_layers=2 --d_layers=1 --factor=3 --enc_in=20 --dec_in=20 --c_out=1 --des=Exp --itr=1 --loss=Sharpe --start_date=2010 --end_date=2015 --limit_asset_number=8

# def _captured_returns_from_all_windows(
#     experiment_name: str,
#     train_intervals: List[Tuple[int, int, int]],
#     volatility_rescaling: bool = True,
#     only_standard_windows: bool = True,
#     volatilites_known: List[float] = None,
#     filter_identifiers: List[str] = None,
#     captured_returns_col: str = "captured_returns",
#     standard_window_size: int = 1,
# ) -> pd.Series:
#     """get sereis of captured returns from all intervals
#
#     Args:
#         experiment_name (str): name of experiment
#         train_intervals (List[Tuple[int, int, int]]): list of training intervals
#         volatility_rescaling (bool, optional): rescale to target annualised volatility. Defaults to True.
#         only_standard_windows (bool, optional): only include full windows. Defaults to True.
#         volatilites_known (List[float], optional): list of annualised volatities, if known. Defaults to None.
#         filter_identifiers (List[str], optional): only run for specified tickers. Defaults to None.
#         captured_returns_col (str, optional): column name of captured returns. Defaults to "captured_returns".
#         standard_window_size (int, optional): number of years in standard window. Defaults to 1.
#
#     Returns:
#         pd.Series: series of captured returns
#     """
#     srs_list = []
#     volatilites = volatilites_known if volatilites_known else []
#     for interval in train_intervals:
#         if only_standard_windows and (
#             interval[2] - interval[1] == standard_window_size
#         ):
#             df = pd.read_csv(
#                 os.path.join(
#                     _get_directory_name(experiment_name, interval),
#                     "captured_returns_sw.csv",
#                 ),
#             )
#
#             if filter_identifiers:
#                 filter = pd.DataFrame({"identifier": filter_identifiers})
#                 df = df.merge(filter, on="identifier")
#             num_identifiers = len(df["identifier"].unique())
#             srs = df.groupby("time")[captured_returns_col].sum() / num_identifiers
#             srs_list.append(srs)
#             if volatility_rescaling and not volatilites_known:
#                 volatilites.append(annual_volatility(srs))
#     if volatility_rescaling:
#         return pd.concat(srs_list) * VOL_TARGET / np.mean(volatilites)
#     else:
#         return pd.concat(srs_list)