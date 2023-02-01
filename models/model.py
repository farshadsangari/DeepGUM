from models import e_step
from models import m_step
import torch.nn as nn
import torch


def EM_algorithm(
    y_preds_train,
    y_trues_train,
    y_preds_val,
    y_trues_val,
    cov,
    prior,
    gamma,
    max_iter_em,
    param_threshold=[1e-8, 1e-8, 1e-8],
):

    Rs = {"R_train": [], "R_val": []}
    for i in range(max_iter_em):
        rn_s_train = e_step(prior, gamma, y_trues_train, y_preds_train, cov)
        Rs["R_train"].append(rn_s_train)
        old_param = [cov, prior, gamma]
        cov, prior, gamma = m_step(prior, rn_s_train, y_trues_train, y_preds_train)
        new_param = [cov, prior, gamma]
        if (
            (new_param[0] - old_param[0]) < param_threshold[0]
            and (new_param[1] - old_param[1]) < param_threshold[1]
            and (new_param[2] - old_param[2]) < param_threshold[2]
        ):
            break

    rn_s_val = e_step(prior, gamma, y_trues_val, y_preds_val, cov)
    Rs["R_val"].append(rn_s_val)

    return cov, prior, gamma, rn_s_train, rn_s_val, Rs


class VGG_LinearRegression(nn.Module):
    def __init__(self):
        super(VGG_LinearRegression, self).__init__()
        self.pretrained_model = torch.hub.load(
            "pytorch/vision:v0.10.0", "vgg16_bn", pretrained=True
        )
        self.LastLinear = nn.Sequential(nn.Linear(in_features=1000, out_features=1))

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.LastLinear(x)
        return x
