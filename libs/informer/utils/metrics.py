import numpy as np
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def class_metric(preds, trues):
    """
    计算多时间步长下的分类指标，并返回包含所有时间步指标的列表 results_list，而不在此处保存 CSV。
    preds: (batch_size, seq_len, time_steps, num_classes) 或 (batch_size, time_steps, num_classes)
    trues: (batch_size, seq_len, time_steps, 1) 或 (batch_size, time_steps, 1)

    返回:
        results_list, 列表中每个元素是形如:
        {
            "Day": t + 1,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    """
    # 1) 获取预测的类别（假设最后一维是 num_classes）
    predicted_classes = np.argmax(preds, axis=-1)  # => (batch_size, seq_len, time_steps)

    # 2) 调整标签形状以便计算评分
    #    假设 trues 原本是 (batch_size, seq_len, time_steps, 1)，去掉最后一维
    true_classes = trues.squeeze(axis=-1)  # => (batch_size, seq_len, time_steps)

    # 3) 准备结果列表
    results_list = []

    # 4) 遍历每个时间步长
    time_steps = true_classes.shape[-1]
    for t in range(time_steps):
        # 选择当前时间步长的预测和真实类别，展平以计算整体指标
        current_predicted_classes = predicted_classes[..., t].reshape(-1)
        current_true_classes = true_classes[..., t].reshape(-1)

        # 计算准确率
        accuracy = (current_predicted_classes == current_true_classes).mean()

        # 计算精确率、召回率和F1分数（macro 平均）
        precision = precision_score(current_true_classes, current_predicted_classes, average='macro', zero_division=0)
        recall = recall_score(current_true_classes, current_predicted_classes, average='macro', zero_division=0)
        f1 = f1_score(current_true_classes, current_predicted_classes, average='macro', zero_division=0)

        # 打印当前时间步长的结果
        print(f"Day {t + 1} - Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        results_list.append(
            {"Day": t + 1, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1})

    return results_list
