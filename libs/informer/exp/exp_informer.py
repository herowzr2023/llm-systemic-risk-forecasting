from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, class_metric
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import time
import warnings

warnings.filterwarnings('ignore')
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
preferred_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = [font for font in preferred_fonts if font in available_fonts] or ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def custom_collate_fn(batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark, date_seq = zip(*batch)
    batch_x = torch.from_numpy(np.stack(batch_x))
    batch_y = torch.from_numpy(np.stack(batch_y))
    batch_x_mark = torch.from_numpy(np.stack(batch_x_mark))
    batch_y_mark = torch.from_numpy(np.stack(batch_y_mark))
    # date_seq 是日期序列的列表，我们保持它的原始格式
    return batch_x, batch_y, batch_x_mark, batch_y_mark, date_seq


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {'informer': Informer, 'informerstack': InformerStack, }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](self.args.enc_in, self.args.dec_in, self.args.c_out, self.args.seq_len,
                self.args.label_len, self.args.pred_len, self.args.factor, self.args.d_model, self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers, self.args.d_ff, self.args.dropout, self.args.attn, self.args.embed, self.args.freq,
                self.args.activation, self.args.output_attention, self.args.distil, self.args.mix, self.device,
                is_class=self.args.is_class, num_class=self.args.num_class).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute, 'WTH': Dataset_Custom, 'ECL': Dataset_Custom, 'Solar': Dataset_Custom,
            'Risk': Dataset_Custom, 'Risk_E': Dataset_Custom, 'custom': Dataset_Custom, 'custom_sen': Dataset_Custom, }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(root_path=args.root_path, data_path=args.data_path, flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len], features=args.features, target=args.target,
            inverse=args.inverse, timeenc=timeenc, freq=freq, cols=args.cols, scaler_type=args.scaler,
            is_class=args.is_class)
        print(flag, len(data_set))
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers,
            drop_last=drop_last, collate_fn=custom_collate_fn)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.is_class:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates) in enumerate(vali_loader):
            pred, true, _ = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                batch_dates)
            if self.args.is_class:
                loss = criterion(pred.detach().cpu().view(-1, self.args.num_class), true.detach().cpu().long().view(-1))
            else:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, _ = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                    batch_dates)
                if self.args.is_class:
                    loss = criterion(pred.view(-1, self.args.num_class), true.long().view(-1))
                else:
                    loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1,
                train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates) in enumerate(test_loader):
            pred, true, _ = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                batch_dates)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        if self.args.is_class is False:
            # 获取预测的类别
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.args.is_class:
            class_metric(preds, trues)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates) in enumerate(pred_loader):
            pred, true, y_date = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                batch_dates)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def evaluate(self, setting, load=False):
        # 如果需要，加载模型
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        # 获取数据集和数据加载器
        datasets = {}
        loaders = {}
        for flag in ['train', 'val', 'test']:
            data, loader = self._get_data(flag=flag)
            datasets[flag] = data
            loaders[flag] = loader

        # 用于保存整体指标；后面会一次性写到 metrics.csv
        metrics_list = []

        # 如果需要记录文件名、轮次等信息
        Filename = getattr(self.args, 'path', "None")
        Epoch = getattr(self.args, 'itr', "0")

        # 依次对 train, val, test 进行评估
        for flag in ['train', 'val', 'test']:
            preds = []
            trues = []
            dates = []

            data = datasets[flag]
            loader = loaders[flag]

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates) in enumerate(loader):
                    pred, true, y_date = self._process_one_batch(data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                        batch_dates)
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

                    # 如果需要时间序列可视化，就把 y_date 收集起来
                    if y_date is not None:
                        # 示例做法：如果有 pred_len，就收集后 pred_len 个日期
                        if hasattr(self.args, 'pred_len'):
                            for date_seq in y_date:
                                dates.extend(date_seq[-self.args.pred_len:])
                        else:
                            # 不确定 pred_len 就全部收集
                            for date_seq in y_date:
                                dates.extend(date_seq)

            # 拼接得到最终预测、真实值
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            dates = np.array(dates)

            # 创建结果文件夹
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # -------------- 分类情形 --------------
            if self.args.is_class:
                # 不用逆变换、不绘图；只计算分类指标
                # 假设 preds.shape = (batch_size, seq_len, time_steps, num_classes)
                # trues.shape = (batch_size, seq_len, time_steps, 1)

                # 调用 class_metric 计算多步分类指标，每步结果都在 results_list
                results_list = class_metric(preds, trues)

                # 这里不在 class_metric 里保存 CSV，而是统一在 evaluate 里保存
                # 1) 先将每个时间步长的指标追加到 metrics_list
                for item in results_list:
                    # 给每条记录加上 文件名、轮次、数据集标识
                    item['文件名'] = Filename
                    item['轮次'] = Epoch
                    item['数据集'] = flag
                    # 例如 item: {Day, Accuracy, Precision, Recall, F1 Score, 文件名, 轮次, 数据集}
                    metrics_list.append(item)

            # -------------- 回归情形 --------------
            else:
                # 如果回归且需要逆变换
                if not self.args.inverse:
                    preds = data.inverse_transform_y(preds)
                    trues = data.inverse_transform_y(trues)

                # 展平/排序并进行可视化保存
                preds_flat = preds.reshape(-1)
                trues_flat = trues.reshape(-1)
                dates_flat = dates.reshape(-1) if len(dates) else []
                forecast_start_flat = []
                horizon_flat = []
                window_id_flat = []
                forecast_start_sorted = []
                horizon_sorted = []
                window_id_sorted = []

                # Preserve the forecast-window identity before sorting by target
                # date.  The legacy file retained only date/true/pred, which made
                # the horizon of duplicate target dates impossible to read
                # directly after np.argsort reordered equal-date observations.
                if len(dates_flat) == len(preds_flat) and len(dates_flat) != 0:
                    output_steps = int(self.args.pred_len)
                    if len(dates_flat) % output_steps == 0:
                        n_windows = len(dates_flat) // output_steps
                        date_matrix = np.asarray(dates_flat).reshape(n_windows, output_steps)
                        forecast_start_flat = np.repeat(date_matrix[:, 0], output_steps)
                        horizon_flat = np.tile(np.arange(1, output_steps + 1), n_windows)
                        window_id_flat = np.repeat(np.arange(n_windows), output_steps)

                if len(dates_flat) == len(preds_flat) and len(dates_flat) != 0:
                    # 按日期排序
                    sorted_indices = np.argsort(dates_flat)
                    dates_sorted = dates_flat[sorted_indices]
                    preds_sorted = preds_flat[sorted_indices]
                    trues_sorted = trues_flat[sorted_indices]
                    if len(horizon_flat):
                        forecast_start_sorted = forecast_start_flat[sorted_indices]
                        horizon_sorted = horizon_flat[sorted_indices]
                        window_id_sorted = window_id_flat[sorted_indices]
                else:
                    # 如果日期和数据点数不匹配，或者没有日期，就原样处理
                    dates_sorted = dates_flat
                    preds_sorted = preds_flat
                    trues_sorted = trues_flat
                    forecast_start_sorted = forecast_start_flat
                    horizon_sorted = horizon_flat
                    window_id_sorted = window_id_flat

                # 保存为 npy
                np.save(folder_path + f'{flag}_pred.npy', preds_sorted)
                np.save(folder_path + f'{flag}_true.npy', trues_sorted)
                np.save(folder_path + f'{flag}_dates.npy', dates_sorted)

                # 如果有日期信息，可以保存到 CSV 并绘图
                if len(dates_sorted) > 0:
                    result_columns = {
                        'date': pd.to_datetime(dates_sorted),
                        'true': trues_sorted,
                        'pred': preds_sorted,
                    }
                    if len(horizon_sorted):
                        result_columns.update(
                            {
                                'forecast_start_date': pd.to_datetime(forecast_start_sorted),
                                'horizon': horizon_sorted,
                                'forecast_window_id': window_id_sorted,
                            }
                        )
                    df = pd.DataFrame(result_columns)
                    df.to_csv(folder_path + f'{flag}_results.csv', index=False)

                    # 绘制预测与真实值对比
                    plt.figure(figsize=(12, 6))
                    plt.plot(dates_sorted, trues_sorted, label='True')
                    plt.plot(dates_sorted, preds_sorted, label='Prediction')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.title(f'{flag.capitalize()} Set Prediction vs True')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(folder_path + f'{flag}_prediction_vs_true.png')
                    plt.close()

                # 计算回归指标
                y_pred = preds_sorted
                y_true = trues_sorted
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                y_true_safe = np.where(y_true == 0, 1e-9, y_true)
                mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
                mspe = np.mean(np.square((y_true - y_pred) / y_true_safe)) * 100

                # 把回归指标也放入 metrics_list（只一条记录即可）
                metrics_list.append(
                    {'文件名': Filename, '轮次': Epoch, '数据集': flag, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                        'MAPE': mape, 'MSPE': mspe})

        # ========== 统一将 metrics_list 存为 CSV ==========
        # 这里为了简化，只存最后一次循环的 folder_path
        # 若想为 train/val/test 分开存，可在循环内分别写
        metrics_df = pd.DataFrame(metrics_list)
        csv_path = os.path.join(folder_path, 'metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dates=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse and (not self.args.is_class):
            outputs = dataset_object.inverse_transform_y(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if batch_dates is not None:
            batch_dates = np.array(batch_dates).reshape(-1, self.args.label_len + self.args.pred_len, 1)
            batch_dates = batch_dates[:, -self.args.pred_len:, f_dim:]
        return outputs, batch_y, batch_dates
