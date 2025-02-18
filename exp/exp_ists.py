import os
import numpy as np
import pandas as pd
import torch
from exp.exp_main import Exp_Main
from utils.metrics import MAE, MSE


class ExpISTS(Exp_Main):

    def test(self, setting, id_array, scalers, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        results_dir = './test_results/' + setting + '/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                y_b, y_t, y_c = batch_y.size()
                dec_inp = torch.zeros(y_b, 1, y_c).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                """outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)"""
                # predizione puntuale
                outputs = outputs[:, -1:, f_dim:]
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1],batch_x.shape[2]))
        #     exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        results_dir = './results/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results = dict()
        results_path = results_dir + self.args.model_id + '.csv'
        if os.path.exists(results_path):
            results = pd.read_csv(results_path, index_col=0).to_dict(orient='index')

        y_pred, y_true = np.reshape(preds, (-1, 1)), np.reshape(trues, (-1, 1))
        y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_true, id_array)])
        y_pred = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                        for y_, id in zip(y_pred, id_array)])
        mse, mae = MSE(y_pred, y_true), MAE(y_pred, y_true)
        print(f'MAE: {mae}, MSE: {mse}')
        
        results[f'E'] = {
            'test_mae': mae, 'test_mse': mse,
            "epoch_times": self.epoch_times, 
            "val_loss": self.val_loss
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.index.name = self.args.model_id
        results.to_csv(results_path)

        return
