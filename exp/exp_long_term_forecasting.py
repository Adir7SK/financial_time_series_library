from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import LogSharpeLoss, SharpeLoss, sharpe_ratio
from utils.metrics import metric
from empyrical import annual_volatility
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


def check_if_this_exact_vector_sequence_already_in_last_n_train_loops(vec, last_vectors, number_rem=20, is_weights=True):
    """
    Check if the given vector is already in the last n vectors.
    :param vec: The vector to check.
    :param last_vectors: The list of last vectors.
    :return: True if the vector is already in the last n vectors, False otherwise.
    """
    found_match = False
    for v in last_vectors:
        found_match = False
        for i in range(len(vec)-1):
            if abs(vec[i+1] - v[i+1]) < 0.001:
                found_match = True
            else:
                found_match = False
                break
        if found_match:
            break

    # if len(last_vectors) and is_in:
    #     print('Got the exact same {!r} as before!!!'.format('weights' if is_weights else 'returns'))
    #     print(vec)

    if len(last_vectors) >= number_rem:
        last_vectors.pop(0)
    last_vectors.append(vec)
    return last_vectors, found_match

def annualized_sharpe_ratio(weights, y_true, periods=252):
    captured_returns = weights * y_true
    mean_returns = torch.mean(captured_returns)
    std_returns = torch.sqrt(
        torch.mean(torch.square(captured_returns))
        - torch.square(mean_returns)
        + 1e-7
    )
    print('Std returns:', std_returns)
    print('Mean returns:', mean_returns)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.recall_last_preds = []
        self.recall_last_returns = []
        self.count_repeated_weights = 0
        self.count_repeated_returns = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='Sharpe'):
        if loss_name == 'Sharpe':
            # Add debugging to SharpeLoss
            class DebugSharpeLoss(SharpeLoss):
                def forward(self, weights, returns):
                    loss = super().forward(weights, returns)

                    # Print intermediate values
                    print("\nLoss computation details:")
                    # print(f"Weights shape: {weights.shape}")
                    # print(f"Returns shape: {returns.shape}")
                    print(f"Captured returns mean: {(weights * returns).mean():.4f}")
                    print(f"Captured returns std: {(weights * returns).std():.4f}")
                    print(f"Final loss: {loss.item():.4f}")

                    return loss

            return DebugSharpeLoss()

        if loss_name == 'Sharpe':
            return SharpeLoss()
        if loss_name == 'LogSharpe':
            return LogSharpeLoss()
        elif loss_name == 'MSE':
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, active_entries) in enumerate(vali_loader):
                if batch_x is None or batch_x.numel() == 0:
                    continue
                active_entries = active_entries.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().to(self.device)
                weights = torch.tanh(pred) * active_entries.unsqueeze(1).unsqueeze(2)
                true = batch_y.detach().cpu()

                loss = criterion(weights.cpu(), true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        def compute_grad_norm():
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5

        # Track loss values and gradients
        loss_values = []
        grad_values = {name: [] for name, _ in self.model.named_parameters()}

        def hook_fn(grad, name):
            if grad is not None:
                grad_values[name].append({
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item()
                })

        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, active_entries) in enumerate(train_loader):
                active_entries = active_entries.float().to(self.device)
                # active_entries = torch.tensor(active_entries).float().to(self.device)
                # print('Length of seq_x:', len(batch_x))
                # print('Length of seq_y:', len(batch_y))
                # print('Batch_size:', self.args.batch_size)

                print('Part of batches for index {}:'.format(i))

                # print(f"Batch_x: {batch_x[0, :5, :5]}")  # Print a small slice of batch_x
                # print(f"Batch_x_mark: {batch_x_mark[0, :5, :5]}")  # Print a small slice of batch_x_mark
                # print(f"Batch_y: {batch_y[0, :5, :5]}")  # Print a small slice of batch_y

                if batch_x is None or batch_x.numel() == 0:
                    continue
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # Generate buy/sell weights (constrained between -1 and 1)
                        weights = torch.tanh(outputs) * active_entries.unsqueeze(1).unsqueeze(2)
                        loss = criterion(weights, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # Generate buy/sell weights (constrained between -1 and 1)
                    print(f'The range of numbers in the predictions is [{outputs.min()}, {outputs.max()}]')
                    print(f'The range in tanh is [{torch.tanh(outputs).min()}, {torch.tanh(outputs).max()}]')
                    weights = torch.tanh(outputs) * active_entries.unsqueeze(1).unsqueeze(2)

                    self.recall_last_preds, weights_seen = check_if_this_exact_vector_sequence_already_in_last_n_train_loops(weights, self.recall_last_preds, number_rem=20, is_weights=True)
                    self.recall_last_returns, returns_seen = check_if_this_exact_vector_sequence_already_in_last_n_train_loops(batch_y, self.recall_last_returns, number_rem=20, is_weights=False)
                    if weights_seen:
                        self.count_repeated_weights += 1
                    if returns_seen:
                        self.count_repeated_returns += 1
                    loss = criterion(weights, batch_y)
                    # for l in train_loss:
                    #     if abs(loss.item() - l) < 0.001:
                    #         print('Got the exact same loss as before!!!')
                    #         annualized_sharpe_ratio(weights, batch_y)

                    train_loss.append(loss.item())

                # Store loss value
                loss_values.append(loss.item())

                override = 10
                if (i + 1) % override == 0: # should be 100 instead of override
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                    # Print recent statistics
                    recent_loss = np.mean(loss_values[-override:])
                    print(f"\nRecent loss mean: {recent_loss:.4f}")

                    # Print gradient statistics for each layer
                    #################################################################
                    for name in grad_values:
                        if grad_values[name]:
                            recent_grads = grad_values[name][-override:]
                            means = [g['mean'] for g in recent_grads]
                            print(f"{name:30} | grad_mean: {np.mean(means):.3e} | grad_std: {np.std(means):.3e}")

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"{name:30} grad mean: {param.grad.mean().item():10.3e}, grad std: {param.grad.std().item():10.3e}")
                    #     else:
                    #         print(f"{name:30} has no gradient")
                    #         print('')


                    #         print("Index: {}".format(i))
                    #         print(f"Batch_x: {batch_x[0, :5, :5]}")  # Print a small slice of batch_x
                    #         print(f"Batch_x_mark: {batch_x_mark[0, :5, :5]}")  # Print a small slice of batch_x_mark
                    #         print(f"Batch_y: {batch_y[0, :5, :5]}")  # Print a small slice of batch_y
                    #         print('')

                    model_optim.step()

            invalid_loss_count = sum(1 for loss_item in train_loss if
                                     loss_item is None or torch.isinf(torch.tensor(loss_item)).any() or torch.isnan(
                                         torch.tensor(loss_item)).any())
            valid_loss_count = len(train_loss) - invalid_loss_count
            print("Invalid loss count: {}, Valid loss count: {}".format(invalid_loss_count, valid_loss_count))

            print("Count of repeated weights: {}, Count of repeated returns: {}".format(self.count_repeated_weights, self.count_repeated_returns))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, active_entries) in enumerate(test_loader):
                if batch_x is None or batch_x.numel() == 0:
                    continue
                # active_entries = active_entries.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = torch.tanh(outputs) * active_entries.unsqueeze(1).unsqueeze(2)
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'

            # mae, mse, rmse, mape, mspe = metric(preds, trues)
            # print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            # f = open("result_long_term_forecast.txt", 'a')
            # f.write(setting + "  \n")
            # f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            # f.write('\n')
            # f.write('\n')
            # f.close()
            #
            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)

            # Sharpe ratio calculation
            buy_sell_vector = torch.tanh(torch.tensor(preds)) # * active_entries.unsqueeze(1).unsqueeze(2)

            # Calculate Sharpe Ratio
            returns = buy_sell_vector.numpy() * trues  # Element-wise multiplication
            volatilites = annual_volatility(returns, period=252)
            # Adjust return values to match the target volatility
            returns = returns*self.args.vol_target / np.mean(volatilites)
            sharpe = sharpe_ratio(returns)

            # Print and save Sharpe Ratio
            print('Sharpe Ratio: {}'.format(sharpe))
            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(setting + "  \n")
                f.write('Sharpe Ratio: {}\n'.format(sharpe))
                f.write('Average Returns: {}\n'.format(np.average(returns)))
                f.write('Average Volatility: {}\n'.format(np.average(volatilites)))
                f.write('\n')

            # Save buy/sell vector and Sharpe Ratio
            np.save(folder_path + 'buy_sell_vector.npy', buy_sell_vector.numpy())
            np.save(folder_path + 'sharpe.npy', np.array([sharpe]))
            # np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return
