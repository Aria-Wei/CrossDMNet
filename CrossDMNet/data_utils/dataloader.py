import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import mne
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import train_test_split

def load_BCI2a_data(data_path, subject, training=True, all_trials=True, st=2, end=4):
    """
    load BCI2a data_utils from .mat files
    return data_return: narray, class_return: narray
    :param data_path:
    :param subject:
    :param training:
    :param all_trials: set to be False to disable trials with artifacts
    :return: data_return: ndarray(nums, channels, seq), class_return: ndarray(classes,)
    """
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6 * 48
    window_length = 7 * 250

    # Define MI-trial window
    fs = 250
    t1 = int(st * fs)  # start time_point
    t2 = int(end * fs)  # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))

    ind = 0

    if training:
        a = sio.loadmat(os.path.join(data_path, f's{subject}', f"A0{subject}T.mat"))
    else:
        a = sio.loadmat(os.path.join(data_path, f's{subject}', f"A0{subject}E.mat"))
    a_data = a['data']
    for ii in range(a_data.size):
        a_data1 = a_data[0, ii][0, 0]
        a_X = a_data1['X']
        a_y = a_data1['y']
        a_trial = a_data1['trial']
        a_artifacts = a_data1['artifacts']

        for trial in range(a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            data_return[ind, :, :] = np.transpose(a_X[int(a_trial[trial]):int(a_trial[trial]) + window_length, :n_channels])
            class_return[ind] = int(a_y[trial, 0])
            ind += 1

    data_return = data_return[0: ind, :, t1:t2]
    class_return = class_return[0: ind]
    class_return = (class_return - 1).astype(int)
    return data_return, class_return


def load_hgd_data(data_path, subject, training=True, pick22chans=True, tmin=0, tmax=2, shuffled=True):
    # tmin=-0.5, tmax=4
    if training:
        edf_path = os.path.join(data_path, 'train', f'{subject}.edf')
    else:
        edf_path = os.path.join(data_path, 'test', f'{subject}.edf')
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_channels([ch for ch in raw.ch_names if 'EEG' in ch])
    new_channel_names = [ch.replace('EEG ', '') for ch in raw.ch_names]
    mapping = {old: new for old, new in zip(raw.ch_names, new_channel_names)}
    raw.rename_channels(mapping)
    if pick22chans:
        eeg_22_channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
        raw.pick_channels(eeg_22_channel_names)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
    raw.set_eeg_reference('average')

    # resample to 250hz
    raw.resample(sfreq=250)

    events, event_ids = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax, preload=True, baseline=None)
    X = epochs.get_data()[..., :-1]
    y = epochs.events[:, 2]
    y = y - 1
    if shuffled:
        X, y = shuffle(X, y, random_state=42)
    return X, y


def load_openBMI_data(data_path, sub_id, training=True, num_samples=500,
                 resample=True, chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]):
    if training:
        data = sio.loadmat(os.path.join(data_path, 'train', f'sess01_subj{sub_id:02d}_EEG_MI.mat'))
    else:
        data = sio.loadmat(os.path.join(data_path, 'test', f'sess02_subj{sub_id:02d}_EEG_MI.mat'))
    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'], data['EEG_MI_test'][0, 0]['smt']), axis=1).astype(
        np.float32)
    y = np.concatenate(
        (data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
        axis=0).astype(int) - 1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()

    x = np.transpose(x, axes=(1, 2, 0)) # -> (n_trials, n_chans, seq_len)
    if resample:
        # resample from 1000HZ to 250HZ
        old_fs, new_fs = s, 250
        n_trials, n_chans, seq_len = x.shape
        new_seq_len = int(seq_len * new_fs / old_fs)
        ori_x = x.copy()
        x = np.zeros((n_trials, n_chans, new_seq_len))
        for i in range(n_trials):
            x[i, ...] = mne.filter.resample(ori_x[i, ...].astype(np.float64), down=old_fs, up=new_fs, npad='auto')

    if chans is not None:
        x = x[:, np.array(chans)]
        c = c[np.array(chans)]

    return x.astype(np.float32)[..., :num_samples], y


def load_data_LOSO(data_path, dataset, subject=None, st=2, end=4, EA=False):
    """
    标准LOSO方法划分数据集，根据subject，将所有其他subject设置为训练集（结合两个session），subject（两个session）设置为测试集
    :param data_path:
    :param dataset:
    :param subject:
    :return:
    """
    n_subjects = 9
    # if dataset == 'HGD':
    #     n_subjects = 14
    if subject is None:
        subject = n_subjects

    X_train = []
    for sub in range(1, n_subjects+1):
        if dataset == 'BCI2a':
            X1, y1 = load_BCI2a_data(data_path, sub, True, st=st, end=end)
            X2, y2 = load_BCI2a_data(data_path, sub, False, st=st, end=end)

        else:
            raise ValueError('dataset must be BCI2a')

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if EA:
            X = euclidean_alignment(X)

        if sub == subject:
            X_test = X
            y_test = y

        elif X_train == []:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    对每个通道进行全局标准化，衡量特定通道在所有实验、所有时间点上的整体信号基线和波动幅度
    :inputs: X_train(nums, C, T)
    :return: X(nums, C, T)
    """
    n_chans = X_train.shape[1]
    for j in range(n_chans):
        mean = np.mean(X_train[:, j, :])
        std = np.std(X_train[:, j, :])
        X_train[:, j, :] = (X_train[:, j, :] - mean) / std
        X_test[:, j, :] = (X_test[:, j, :] - mean) / std
    return X_train, X_test


def get_data(data_path,
             subject=None,
             dataset='BCI2a',
             LOSO=False,
             is_standardized=True,
             is_shuffle=True,
             is_one_hot=False,
             st=2,
             end=4,
             EA=False
             ):
    """
    :return: X_train(nums, C, T), X_test, y_train, y_test
    """
    if LOSO:
        X_train, X_test, y_train, y_test = load_data_LOSO(data_path, dataset=dataset, subject=subject, st=st, end=end, EA=EA)
    else:
        if dataset == 'BCI2a':
            X_train, y_train = load_BCI2a_data(data_path, subject, True, st=st, end=end)
            X_test, y_test = load_BCI2a_data(data_path, subject, False, st=st, end=end)
        elif dataset == 'HGD':
            X_train, y_train = load_hgd_data(data_path, subject, True)
            X_test, y_test = load_hgd_data(data_path, subject, False)
        elif dataset == 'openBMI':
            X_train, y_train = load_openBMI_data(data_path, subject, True)
            X_test, y_test = load_openBMI_data(data_path, subject, False)
        elif dataset == 'GigaDB':
            X, y = load_Giga_data(data_path, subject)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                shuffle=True,
                                                                random_state=42,
                                                                test_size=0.2)
        else:
            raise Exception(f'{dataset} not supported')

    if is_shuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    if is_standardized:
        X_train, X_test = standardize_data(X_train, X_test)

    if is_one_hot:
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test


def euclidean_alignment(data):
    """
    对 EEG 数据进行欧几里得对齐 (EA)。

    Args:
        data (np.ndarray): 形状为 (N_trials, C_channels, T_timepoints) 的数据。

    Returns:
        np.ndarray: 对齐后的数据，形状保持不变。
    """
    # 1. 检查维度
    if data.ndim != 3:
        raise ValueError("Data must be 3D: (trials, channels, time)")

    N, C, T = data.shape

    # 2. 计算每个 trial 的协方差矩阵
    # 居中数据通常是必要的，但如果数据已经带通滤波过（均值为0），这一步可以简化
    # 这里我们计算样本协方差: R = (X * X.T) / (T - 1)
    cov_matrices = []
    for i in range(N):
        trial = data[i]
        cov = np.dot(trial, trial.T) / (T - 1)
        cov_matrices.append(cov)

    cov_matrices = np.array(cov_matrices)

    # 3. 计算参考矩阵 (Reference Matrix) -> 平均协方差
    ref_cov = np.mean(cov_matrices, axis=0)

    # 4. 计算变换矩阵 P = R^(-1/2)
    # 使用 fractional_matrix_power 计算矩阵的 -0.5 次幂
    try:
        projection_matrix = fractional_matrix_power(ref_cov, -0.5)
    except Exception as e:
        print("Error in matrix inversion, adding regularization...")
        # 如果矩阵奇异，加一点微小的正则化项
        reg = np.eye(C) * 1e-6
        projection_matrix = fractional_matrix_power(ref_cov + reg, -0.5)

    # 5. 应用变换到每个 trial: X_new = P * X
    # 利用 Einstein summation 简化批量矩阵乘法
    # 'ij,njk->nik' 意为: P(i,j) * Data(n,j,k) -> NewData(n,i,k)
    # i, j 是通道维度，n 是样本维度，k 是时间维度
    data_aligned = np.einsum('ij,njk->nik', projection_matrix, data)

    return data_aligned.astype(np.float32)


def load_Giga_data(data_path, subject, all_trials=True, pick22chans=True, st=0.0, end=3.0):
    """
    加载 GigaDB (Cho2017) 数据集
    包含：通道挑选、去均值、单位缩放、事件打标切分(Epochs)、下采样(250Hz)。
    取0-2s的数据（0-3s）
    """
    fs_original = 512  # GigaDB 原始采样率
    fs_target = 250  # 目标下采样率

    # 1. 定义 GigaDB 64 个脑电通道的严格顺序
    giga_64_channels = [
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
        "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
        "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
        "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
        "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
        "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2"
    ]

    # 2. 确定需要提取的通道索引 (抛弃 64 以后的 EMG 肌电通道)
    if pick22chans:
        target_channels = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
        ]
    else:
        target_channels = giga_64_channels

    chan_indices = [giga_64_channels.index(ch) for ch in target_channels if ch in giga_64_channels]

    # 加载 .mat 文件
    file_name = f"s{int(subject):02d}.mat"
    file_path = os.path.join(data_path, file_name)
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    eeg_struct = mat_data['eeg']

    # 3. 提取 2D 连续数据并进行清洗
    # 仅提取所需通道，当前形状: (Channels, 358400)
    X_left_2d = eeg_struct.imagery_left[chan_indices, :]
    X_right_2d = eeg_struct.imagery_right[chan_indices, :]

    # 对齐官方：去除直流偏移 (减去均值) + 转换单位 (微伏转为伏特)
    X_left_2d = (X_left_2d - np.mean(X_left_2d, axis=1, keepdims=True)) * 1e-6
    X_right_2d = (X_right_2d - np.mean(X_right_2d, axis=1, keepdims=True)) * 1e-6

    # 4. 提取事件打标通道，用于定位 Trial 的起始点
    events = eeg_struct.imagery_event.flatten()
    # 寻找事件上升沿 (0突变为非0的索引，即提示 Cue 出现的瞬间)
    onsets = np.where((events[:-1] == 0) & (events[1:] != 0))[0] + 1
    if len(onsets) == 0:
        onsets = np.nonzero(events)[0]  # 兼容保护处理

    # 5. 时间窗口计算与切片 (Epoching)
    t1_idx = int(st * fs_original)
    t2_idx = int(end * fs_original)

    def extract_epochs(data_2d, onsets, t1, t2):
        epochs = []
        for onset in onsets:
            start = onset + t1
            end = onset + t2
            # 防止切片越界
            if start >= 0 and end <= data_2d.shape[1]:
                epochs.append(data_2d[:, start:end])
        return np.stack(epochs, axis=0)

    # 将 2D 数据切分为 3D 的 (Trials, Channels, Time)
    X_left_3d = extract_epochs(X_left_2d, onsets, t1_idx, t2_idx)
    X_right_3d = extract_epochs(X_right_2d, onsets, t1_idx, t2_idx)

    # 生成标签: 0 代表 left hand, 1 代表 right hand
    y_left = np.zeros(X_left_3d.shape[0], dtype=int)
    y_right = np.ones(X_right_3d.shape[0], dtype=int)

    data_return = np.concatenate((X_left_3d, X_right_3d), axis=0)
    class_return = np.concatenate((y_left, y_right), axis=0)

    # 6. 剔除伪影 (Bad Trials)
    if not all_trials:
        try:
            if hasattr(eeg_struct, 'bad_trial_indices'):
                bad_indices = np.array(eeg_struct.bad_trial_indices) - 1
                # 越界保护 (确保 bad_indices 在当前有效 trials 范围内)
                bad_indices = bad_indices[bad_indices < data_return.shape[0]]

                valid_mask = np.ones(data_return.shape[0], dtype=bool)
                valid_mask[bad_indices] = False

                data_return = data_return[valid_mask]
                class_return = class_return[valid_mask]
        except Exception as e:
            print(f"Warning: Failed to filter artifacts for subject {subject}. Exception: {e}")

    # 7. 下采样逻辑 (512Hz -> 250Hz)
    original_seq_len = data_return.shape[2]
    target_seq_len = int(original_seq_len * fs_target / fs_original)
    # 沿时间轴 (-1) 执行 FFT 重采样
    data_return = resample(data_return, target_seq_len, axis=-1)

    return data_return, class_return

