# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
from finance_collection.constants import trd_days_year


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class SharpeLoss(nn.Module):
    """
    This class/layer will only be used as loss function at training for the optimization, hence it is to be minimized.
    Therefore, we are negating the Sharpe ratio here. For validation we use positive Sharpe ratio.
    """
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, weights, y_true):
        """
        Note the comment under the class' signature.
        Calc Sharpe ratio only the standard deviation is calculated manually with addition of a small epsilon, i.e.
        std = Sqrt((SUM(x^2)-SUM(x)^2 + epsilon)/N), where x are the inputs and N is the number of inputs
        :param y_true: Sequence of returns
        :param weights: The weight of each corresponding asset in our portfolio
        :return: (int) Sharpe Ratio
        """
        captured_returns = weights * y_true
        mean_returns = t.mean(captured_returns)
        std_returns = t.sqrt(
            t.mean(t.square(captured_returns))
            - t.square(mean_returns)
            + 1e-7
        )
        if t.isnan(std_returns) or t.isinf(std_returns):
            raise ValueError("Standard deviation of returns is NaN or infinite.")
        return -(
                mean_returns
                / std_returns
                * t.sqrt(t.tensor(trd_days_year))
        )


def sharpe_ratio(returns, risk_free=0.0, periods=252):
    """
    Calculate the annualized Sharpe ratio. This is used for test and validation.

    Args:
        returns: Array-like of returns
        risk_free: Risk-free rate (default 0)
        periods: Number of periods in a year (252 for daily raw_data_prep_before_feature_construction, assuming trading days)

    Returns:
        float: Annualized Sharpe ratio
    """
    returns = np.array(returns)
    # print("The first 10 returns are: ", returns[:10])
    # print("The shape of returns is: ", returns.shape)
    excess_returns = returns - risk_free
    # print("The first 10 excess_returns are: ", excess_returns[:10])
    # print("The shape of excess_returns is: ", excess_returns.shape)
    if len(excess_returns) < 2:
        raise ValueError(f"The length of returns is: {len(returns)} causing the Sharpe Ratio to be NaN")
        return np.nan

    mean_excess_return = np.mean(excess_returns)
    mean_excess_return = np.nanmean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    num_nans = np.isnan(excess_returns).sum()
    std_excess_return = np.nanstd(excess_returns, ddof=1)

    if std_excess_return == 0 or np.isnan(std_excess_return):
        # print("The first 10 returns are: ", returns[:10])
        # print("The first 10 excess returns are: ", excess_returns[:10])
        # print("The number of NaNs in excess returns is: ", num_nans)
        # print("The standard deviation without NaNs is: ", std_excess_return)
        raise ValueError("The standard deviation of excess returns is zero or NaN")

    if std_excess_return == 0:
        return np.nan

    sharpe = mean_excess_return / std_excess_return
    # print("You reached the end of the calculation, the mean is: ", mean_excess_return)
    # print("You reached the end of the calculation, the std is: ", std_excess_return)
    # print("You reached the end of the calculation, the Sharpe ratio is: ", sharpe)
    return np.sqrt(periods) * sharpe
