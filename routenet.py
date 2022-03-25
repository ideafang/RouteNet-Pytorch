import pytorch_lightning as pl
import torch
import torch.nn as nn


class RouteNet(pl.LightningModule):
    def __init__(self, dim_link, dim_path, dim_linear, t):
        super().__init__()
        self.t = t
        self.link_embedding = nn.Linear(1, dim_link)
        self.path_embedding = nn.Linear(1, dim_path)

        self.path_update = nn.GRU(dim_link, dim_path, batch_first = True)
        self.edge_update = nn.GRUCell(dim_path, dim_link)

        self.readout = nn.Sequential(
            nn.Linear(dim_path, dim_linear),
            nn.Linear(dim_linear, dim_linear),
            nn.Linear(dim_linear, 1)
        )

        self.loss = nn.MSELoss(reduction='mean')
        self.mae_calc = nn.L1Loss()

    def forward(self, x):
        # batch_size = 1
        bw = x['bandwidth'].view(-1, 1)
        pk = x['package'].view(-1, 1)
        link_state = self.link_embedding(bw)  # [Ne, De]
        path_state = self.path_embedding(pk).unsqueeze(0)  # [1, Np, Dp]
        path = x['path'].squeeze()
        row = x['row'].squeeze()
        col = x['col'].squeeze()
        link_idx = x['link_idx'].squeeze()

        for _ in range(self.t):
            path_matrix = link_state[path]  # [Np, Lp, De]
            output, path_state = self.path_update(path_matrix, path_state)
            # output [Np, Lp, Dp]  path_state [1, Np, Dp]
            m = output[row, col]  # [Nw, Dp]
            x = torch.zeros_like(link_state)
            x.index_add_(0, link_idx, m)  # [Ne, Dp]
            link_state = self.edge_update(x, link_state)  # [Ne, De]
        
        r = self.readout(path_state).squeeze(-1)
        return r

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    # calc pearsonr
    def pearsonr(self, y_pred, y_true):
        # y_pred: [batch, seq]
        # y_true: [batch, seq]
        y_pred, y_true = y_pred.view(-1), y_true.view(-1)
        centered_pred = y_pred - torch.mean(y_pred)
        centered_true = y_true - torch.mean(y_true)
        covariance = torch.sum(centered_pred * centered_true)
        bessel_corrected_covariance = covariance / (y_pred.size(0) - 1)
        std_pred = torch.std(y_pred, dim=0)
        std_true = torch.std(y_true, dim=0)
        corr = bessel_corrected_covariance / (std_pred * std_true)
        return corr
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # y, y_hat = y.squeeze(), y_hat.squeeze()
        mae = self.mae_calc(y_hat, y)
        rho = self.pearsonr(y_hat, y)
        mre = torch.mean(torch.abs(y - y_hat) / torch.abs(y))
        return loss, mae, rho, mre

    def validation_step(self, batch, batch_idx):
        loss, mae, rho, mre = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "mae": mae, "rho": rho, "mre": mre}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, mae, rho, mre = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, "mae": mae, "rho": rho, "mre": mre}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self(x)
        return y_hat
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)