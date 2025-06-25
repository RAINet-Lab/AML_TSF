import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import shap
import h5py
import polars as pl
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import torch
import polars as pl
import argparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from models import LSTM, DLinear, PatchTST,NeuroFlexMLP,Linear,TSMixer,PatchTST_c,Linear_c,TSMixer_c,MultiRocket

def clean_outliers(X,per=.95):
    '''X: weight matrix for a single index in the prediction dims= [B,L]'''
    percentiles = torch.quantile(torch.abs(X), per, dim=0)
    means = torch.mean(X, dim=0)
    mask = torch.abs(X) > percentiles
    X[mask] = means.expand_as(X)[mask]
    return X

def rescale_11(array):
    """Rescale "array" into the range [-1,1].
    
    Parameters
    _______________
    
    array: np.array
    Array to rescale
    
    """
    return (array-array.min()+array-array.max())/array.ptp()

def correlation_matrix(mat, arr):
    """Expand function corr2_coeff for several windows at the same time. Inputs would be GASF or GADF and the scores vector from either LRP or SHAP.
    
    Parameters
    _______________
    
    mat: np.array
    Array that contains several matrix windows. Shape (ner_windows, n, n)
    
    arr: np.array
    Array that contains several array windows. Shape (ner_windows, n)
    
    """
    
    if len(mat.shape) < 3: raise ValueError('Expected mat shape (ner_windows, n, n), provided {}'.format(mat.shape))
    if len(arr.shape) < 2: raise ValueError('Expected arr shape (ner_windows, n), provided {}'.format(arr.shape))
    
    if arr.shape[0] != mat.shape[0]:
        raise ValueError('mat and arr must have the same number of windows,'+
                         ' given {} (mat) and {} (arr)'.format(mat.shape[0], arr.shape[0]))
    
    corr_matrix = np.empty((arr.shape[0], arr.shape[1],1))
    
    for i in range(arr.shape[0]):
    
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = mat[i] - mat[i].mean(1)[:, None]
        B_mB = arr[i, None] - arr[i, None].mean(1)[:, None]
        
        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)


        # Finally get corr coeff
        corr_matrix[i] = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
        
    return corr_matrix

def create_dataset(dataset, lookback,lookfront=1,backend='torch'):

    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the channel, second is the batch_size
        lookback: Size of window for prediction
        lookfront: Size of the output. Number of future time steps 
        backend: torch or numpy
    """
    X= np.zeros(shape=[(dataset.shape[1]-lookback-lookfront),dataset.shape[0],lookback])
    y=np.zeros(shape=[(dataset.shape[1]-lookback-lookfront),dataset.shape[0],lookfront])
    for i in range(dataset.shape[1]-lookback-lookfront):
        feature = dataset[:,i:i+lookback]
        target = dataset[:,i+lookback:i+lookback+lookfront]
        X[i]=feature
        y[i]=target
    if backend=='torch':
        return torch.tensor(X,dtype=torch.float), torch.tensor(y,dtype=torch.float)
    else:
        return X, y
def plot_corr_matrix(corr_matrix, start_end, history, plot = True, save = False, path = None, model = None, high_q = False,latex=False):
    
    """
    Plot the correlation matrix obtained with the previous function as a colormap.
    
    Parameters
    _______________
    
    corr_matrix: np.array
    Array to plot as a colormap. Shape (ner_windows, history).
    
    start_end: list or tuple
    List or tuple that contains the range of the matrix to plot, with elements "start" and "end". The range that the function plots is [start, end).
    
    history: int
    Size of the windows we use. If different from 20, please reset the yticks.
    
    plot: bool, default "True"
    Select if plot or not the matrix.
    
    save: bool, default "False"
    Select if save the plot as an image.
    
    path: str
    If "save" is "True", path contains the path where the image is saved.
    
    model: str
    Name of the AI model we used to construct the correlation matrix.
    
    high_q: bool, default "False"
    If "True", set dpi to 1200. Else, dpi is 300.
    
    """
    
    start, end = start_end
    
    corr_matrix = np.squeeze(corr_matrix[start:end]).T[::-1]
    
    # Plot matrix of correlation as a colormap

    fig = plt.figure(figsize=(30,10))
    im = plt.imshow(corr_matrix, interpolation='None', cmap=plt.cm.PiYG)#, vmin = -1.0, vmax = 1.0) #aspect='equal')
    cbar=plt.colorbar(im, fraction = 0.01)
    cbar.ax.tick_params(labelsize=17)
    
    #plt.title('Pearson coefficients vector for each window', fontsize = 17)
    plt.ylabel('Window of n = {} samples'.format(history), fontsize = 25)
    plt.xlabel('Window', fontsize = 25)

    #plt.yticks([0, 4, 9, 14, 19], [0, 5, 10, 15, 19][::-1], fontsize = 22)
    #plt.xticks(list(range(0, end-start,10))+[end-start-1], list(range(start, end, 10))+[end-1], fontsize = 22)

    if high_q: dpi=1200
    else: dpi = 300
    
    if latex==True:
        tikzplotlib.save('{}/correlation_matrix_model_{}_windows_{}_{}.tex'.format(path, model, start, end))
        
    
    if save: plt.savefig('{}/correlation_matrix_model_{}_windows_{}_{}.png'.format(path, model, start, end), dpi=dpi)
    if plot: plt.show()
    plt.close(fig)
def compare_window2(start,end,X_test,X_gasf, score,corr_matrix, history, plot=True, save = False, path=None, high_q = False,lim=[0,100]):
    
    """
    This function is used to plot (and/or save) the time series (first row), the LRP or SHAP score (2nd row), the GASF (3rd row) and the correlation vector (4th row) for the windows between "start" and "end". 
    
    We do not recomend to try to plot more than 7 or 8 windows for the sake of visualization quality.
    
    Parameters
    _______________
    
    start: int
    First window to plot.
    
    end:int
    Last window to plot.
    
    X_test: np.array
    Array that contains the windows of the time series we want to display. Shape (ner_windows, history).
    
    X_gasf: np.array
    Array that contains the GAF of the windows. Shape (ner_windows, history, history).
    
    score: np.array
    Array that contains the XAI method scores of the windows. Shape (ner_windows, history).
    
    corr_matrix: np.array
    Array that contains the correlation vectors of the windows. Shape (ner_windows, history).
    
    history: int
    Size of the windows we use.
    
    plot: bool, default "True"
    Select if plot or not the matrix.
    
    save: bool, default "False"
    Select if save the plot as an image.
    
    path: str
    If "save" is "True", path contains the path where the image is saved.
    
    high_q: bool, default "False"
    If "True", set dpi to 1200. Else, dpi is 300.
    
    """
    
    windows = range(start,end+1)
    score = rescale_11(score)

    fig = plt.figure(figsize=(6*len(windows),5*len(windows)))
    axes= fig.subplots(4, len(windows), sharey='row')#, sharex='col')

    for i, e in enumerate(windows):
        axes[0,i].plot(X_test[e])
        #axes[0,i].set_ylim(lim[0],lim[1])
        axes[0,i].set_title('Window {}'.format(e),fontsize=25)
        axes[0,i].scatter(X_test[e].argmin(), X_test[e].min(), color='b')
        axes[0,i].scatter(X_test[e].argmax(), X_test[e].max(), color='r')
        #axes[0,0].set_yticks(range(0,101,20))
        #axes[0,0].set_yticklabels(range(0,101,20), fontsize=22)

        axes[1,i].plot(score[e])
        axes[1,i].set_ylim([-1,1])
        axes[1,0].set_yticks(np.arange(-1,1.1,.25))
        axes[1,0].set_yticklabels(np.arange(-1,1.1,.25), fontsize=22)


        axes[2,i].imshow(X_gasf[e], cmap='rainbow')
        #axes[2,0].set_yticks([0, 5, 10, 15, 19])
        #axes[2,0].set_yticklabels([0, 5, 10, 15, 19], fontsize=22)


        axes[3,i].plot(corr_matrix[e])
        axes[3,i].set_ylim([-1,1])
        axes[3,0].set_yticks(np.arange(-1,1.1,.25))
        axes[3,0].set_yticklabels(np.arange(-1,1.1,.25), fontsize=22)


        axes[0,0].set_ylabel('Window of n = {}\n samples (time points)'.format(history), fontsize=25)
        axes[1,0].set_ylabel('SHAP values', fontsize=25)
        axes[2,0].set_ylabel('GASF\n\n', fontsize=25)
        axes[3,0].set_ylabel('Correlation vector', fontsize=25)
    
    #for ax in axes.flatten():
        #ax.set_xticks([0, 5, 10, 15, 19])
        #ax.set_xticklabels([0, 5, 10, 15, 19], fontsize=22)
    
    if high_q: dpi=1200
    else: dpi = 'figure'
        
    if save: plt.savefig(path+'/window_range_{}_{}.png'.format(start, end), dpi=dpi)

def load_raw_data(dataset_config, dataset):
    
    if 'PEMS' in dataset:
        raw_data = np.load(dataset_config['data_filename'])['data']
        raw_data = raw_data[:,0,:]
        train_data_seq = raw_data[:int(0.6 * raw_data.shape[0])]
        val_data_seq = raw_data[int(0.6 * raw_data.shape[0]):int(0.8 * raw_data.shape[0])]
        test_data_seq = raw_data[int(0.8 * raw_data.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq

    elif dataset == 'ETTm1' or dataset == 'Weather':
        raw_data = pd.read_csv(dataset_config['data_filename'])
        raw_data_feats = raw_data.values[:, 1:]
        raw_data_stamps = raw_data.values[:, 0]
        raw_data_stamps = pd.to_datetime(raw_data_stamps)

        # raw_data_stamps = raw_data_stamps.to_numpy()

        train_data_seq = raw_data_feats[:int(0.6 * raw_data_feats.shape[0])]
        val_data_seq = raw_data_feats[int(0.6 * raw_data_feats.shape[0]):int(0.8 * raw_data_feats.shape[0])]
        test_data_seq = raw_data_feats[int(0.8 * raw_data_feats.shape[0]):]

        train_data_stamps = raw_data_stamps[:int(0.6 * raw_data_stamps.shape[0])]
        val_data_stamps = raw_data_stamps[int(0.6 * raw_data_stamps.shape[0]):int(0.8 * raw_data_stamps.shape[0])]
        test_data_stamps = raw_data_stamps[int(0.8 * raw_data_stamps.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps

    elif dataset == 'euma':
        df=pl.read_csv(dataset_config['data_filename'])
        Y_df=df.select(
            ((pl.col('vol_up')+pl.col('vol_dn')).alias('y')),
            ((pl.col('date_time').str.to_datetime("%Y-%m-%d %H:%M:%S")).alias('ds')),
            (pl.col('new_name').alias('unique_id'))
        )
        cutoff_date_beginning = pl.datetime(2019, 4, 1, 0, 0, 0)
        cutoff_date_0_0_0 = pl.datetime(2019, 3, 31)
        cutoff_date = pl.datetime(2019, 6, 1, 0, 0, 0)

        Y_df = Y_df.filter(
            ~(
                (pl.col("ds").dt.truncate("1d") > cutoff_date_0_0_0) & 
                (pl.col("ds").dt.time() == pl.time(0, 0, 0))
            )
        )
        Y_df = Y_df.filter(
            ~(
                (pl.col("ds").dt.truncate("1d") >= cutoff_date)
            )
        )
        Y_df = Y_df.filter(
            ~(
                (pl.col("ds").dt.truncate("1d") < cutoff_date_beginning)
            )
        )
        Y_df=Y_df.sort('ds')
        Y_df=Y_df.group_by_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))  #Aggregate all of the applications
        Y_df=Y_df[:-1200].to_pandas()
        n_time = len(Y_df['ds'].unique())
        val_size = int(.2 * n_time)
        test_size = int(.2 * n_time)
        scaler=StandardScaler()
        scaler.fit(np.array(Y_df['y'].iloc[:-test_size]).reshape(-1,1))
        Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
        n_series = len(Y_df.unique_id.unique())
        time_series=np.array(Y_df['y']).reshape(-1, n_series, 1)
        train_data_seq = time_series[:int(0.6 * time_series.shape[0])]
        val_data_seq = time_series[int(0.6 * time_series.shape[0]):int(0.8 * time_series.shape[0])]
        test_data_seq = time_series[int(0.8 * time_series.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0))[0][0]
        train_std = np.std(train_data_seq, axis=(0))[0][0]
            
        return train_mean, train_std, train_data_seq, test_data_seq

    else:
        raise ValueError('Dataset not supported')

def load_results_dict(h5f):
    results_dict = {}
    for dataset_name in h5f.keys():
        dataset_grp = h5f[dataset_name]
        results_dict[dataset_name] = {}

        for model_name in dataset_grp.keys():
            model_grp = dataset_grp[model_name]
            results_dict[dataset_name][model_name] = {}
            
            if "model_parameters" in model_grp:
                model_params_grp = model_grp["model_parameters"]
                results_dict[dataset_name][model_name]["model_parameters"] = {
                    param_name: torch.tensor(model_params_grp[param_name][...]) 
                    for param_name in model_params_grp.keys()
                }
            for attack in model_grp.keys():
                if attack == "model_parameters":
                    continue
                attack_grp = model_grp[attack]
                results_dict[dataset_name][model_name][attack] = {}

                for key in attack_grp.keys():
                    if key == "dyn_decision_tensor":
                        dyn_tensor_grp = attack_grp[key]
                        results_dict[dataset_name][model_name][attack][key] = [
                            torch.tensor(dyn_tensor_grp[f'tensor_{idx}'][...]) for idx in range(len(dyn_tensor_grp.keys()))
                        ]
                    else:
                        results_dict[dataset_name][model_name][attack][key] = torch.tensor(attack_grp[key][...])

    return results_dict

def load_config(h5f):
    out_dict = {}
    for key, item in h5f.items():
        if isinstance(item, h5py.Group):
            out_dict[key] = load_config(item)
        else:
            data = item[()]
            if isinstance(data, bytes):
                out_dict[key] = data.decode('utf-8')
            elif isinstance(data, np.ndarray):
                if data.ndim == 0:
                    out_dict[key] = data.item()
                else:
                    out_dict[key] = data
            else:
                out_dict[key] = data
    return out_dict


def load_euma(lookback, horizon):
    df = pl.read_csv('./dataset/euma.csv')

    Y_df = df.select(
        ((pl.col('vol_up') + pl.col('vol_dn')).alias('y')),
        ((pl.col('date_time').str.to_datetime("%Y-%m-%d %H:%M:%S")).alias('ds')),
        (pl.col('new_name').alias('unique_id'))
    )
    # print(Y_df)
    cutoff_date_beginning = pl.datetime(2019, 4, 1, 0, 0, 0)
    cutoff_date_0_0_0 = pl.datetime(2019, 3, 31)
    cutoff_date = pl.datetime(2019, 6, 1, 0, 0, 0)

    Y_df = Y_df.filter(
        ~(
            (pl.col("ds").dt.truncate("1d") > cutoff_date_0_0_0) & 
            (pl.col("ds").dt.time() == pl.time(0, 0, 0))
        )
    )
    Y_df = Y_df.filter(
        ~(
            (pl.col("ds").dt.truncate("1d") >= cutoff_date)
        )
    )
    Y_df = Y_df.filter(
        ~(
            (pl.col("ds").dt.truncate("1d") < cutoff_date_beginning)
        )
    )

    # print(Y_df)
    Y_df = Y_df.sort("ds")
    Y_df=Y_df.group_by_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))  #Aggregate all of the applications
    Y_df=Y_df[:-1200].to_pandas()
    n_time = len(Y_df['ds'].unique())
    val_size = int(.2 * n_time)
    test_size = int(.2 * n_time)
    scaler=StandardScaler()
    scaler.fit(np.array(Y_df['y'].iloc[:-test_size]).reshape(-1,1))
    Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
    n_series = len(Y_df.unique_id.unique())
    time_series=np.array(Y_df['y']).reshape(n_series, -1)
    X,y=create_dataset(time_series,lookback,horizon)
    X,y=X.permute(0,2,1),y.permute(0,2,1)
    X_train,y_train=X[:-test_size],y[:-test_size]
    X_test,y_test=X[-test_size:],y[-test_size:]
    return X_train,X_test,y_train,y_test

def load_users(lookback,horizon,frequency=3,**kwargs):
    df=pl.read_csv('./dataset/users_allBS.csv')
    df=df.filter(pl.col('frequency')==frequency)

    Y_df=df.select(
        (pl.col('user_unique').alias('y')),
        (pl.from_epoch("timestamp", time_unit="s").alias('ds')),
        (pl.col('frequency').alias('unique_id'))
    )
    Y_df=Y_df.sort('ds')
    Y_df=Y_df.group_by_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))
    Y_df=Y_df.to_pandas()
    n_time = len(Y_df['ds'].unique())
    val_size = int(.1 * n_time)
    test_size = int(.2 * n_time)
    scaler=StandardScaler()
    scaler.fit(np.array(Y_df['y'].iloc[:-test_size]).reshape(-1,1))
    Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
    n_series = len(Y_df.unique_id.unique())
    time_series=np.array(Y_df['y']).reshape(n_series, -1)
    X,y=create_dataset(time_series,lookback,horizon)
    X,y=X.permute(0,2,1),y.permute(0,2,1)
    X_train,y_train=X[:-test_size],y[:-test_size]
    X_test,y_test=X[-test_size:],y[-test_size:]

    return X_train,X_test,y_train,y_test

def load_PEMS03(dataset_config):
    lookback=dataset_config['loopback']
    horizon=dataset_config["horizon"]
    file = dataset_config["data_filename"]
    data = np.load(file)
    sensor = 0
    # print('data',data)
    df = pl.DataFrame({
            'sensor': data['data'][:,sensor].squeeze(),
    })
    # print('df',df)
    date_range = pd.date_range(start="2018-09-01", end="2018-11-30 23:55:00", freq="5min")
    date_series = pl.Series("date_time", date_range)
    df = df.with_columns(date_series)
    # print('df after',df)
    Y_df=df.select(
        ((pl.col('sensor')).alias('y')),
        ((pl.col('date_time')).alias('ds'))
    )
    # print('Ydf',Y_df)
    Y_df=Y_df.sort('ds')
    Y_df=Y_df.group_by_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))  #Aggregate all of the applications
    # print('Ydf',Y_df)
    Y_df=Y_df[:-1200].to_pandas()
    # print('Ydf',Y_df)
    n_time = len(Y_df['ds'].unique())
    # print('ntime', n_time)
    val_size = int(.2 * n_time)+1
    atk_size = int(.2 * n_time)
    scaler=StandardScaler()
    scaler.fit(np.array(Y_df['y'].iloc[:-val_size]).reshape(-1,1))
    Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
    # print('Ydf',Y_df)
    n_series = len(Y_df.unique_id.unique())
    time_series=np.array(Y_df['y']).reshape(n_series, -1)
    # print('time_series', time_series)
    X,y=create_dataset(time_series,lookback,horizon)
    X,y=X.permute(0,2,1),y.permute(0,2,1)
    X_train,y_train=X[:-val_size],y[:-val_size]
    X_test,y_test=X[-val_size:-atk_size],y[-val_size:-atk_size]
    X_atk, y_atk = X[-atk_size:], y[-atk_size:]

    return X_train,X_test,X_atk,y_train,y_test,y_atk

def load_UCR(lookback,horizon,dataset,**kwargs):
    data_dict={'earthquakes':'Earthquakes','italypd':'ItalyPowerDemand','chinatown':'Chinatown','plane':'Plane','yoga':'Yoga','eumaclf':'EUMA','computers':'Computers'}
    name=data_dict[dataset]
    train,test=np.loadtxt(f'./dataset/{name}/{name}_TRAIN.tsv',delimiter='\t'),np.loadtxt(f'./dataset/{name}/{name}_TEST.tsv',delimiter='\t')
    scaler=StandardScaler()
    scaler.fit(train[:,1:])
    X_train,y_train=torch.tensor(scaler.transform(train[:,1:])).float(),torch.tensor(train[:,0]).long()
    X_test,y_test=torch.tensor(scaler.transform(test[:,1:])).float(),torch.tensor(test[:,0]).long()
    if y_train.min()>0:
        y_train=y_train-1
        y_test=y_test-1
    return X_train.unsqueeze(dim=-1),X_test.unsqueeze(dim=-1),y_train,y_test

def load_basic(lookback, horizon, dataset, scaler=StandardScaler(), split=[0.8, 0.0, 0.2] ,univariate= 'True',**kwargs):
    '''Generate tensors for training, testing, and validating.
        dataset: path to dataset
        lookback: window size
        horizon: prediction length
        scaler: scaling class to use
        split: train-val-test split
    '''
    dataset_dict={'ili': './dataset/national_illness.csv','etth2':'./dataset/ETTh2.csv','Vtraffic':'./dataset/Vtraffic.csv','synthetic':'./dataset/synthetic.csv','cperiod':'./dataset/cperiod.csv'}
    dataset=dataset_dict[dataset]
    # Check if split size is 3
    if len(split) != 3:
        raise ValueError("Size of train-val-test split must be 3")
    
    # Check if the sum of split elements is equal to 1
    if sum(split) != 1:
        raise ValueError("The sum of train-val-test elements must be equal to 1")
    
    data = pd.read_csv(dataset).values[:, 1:]
    train_size, test_size = int(split[0] * data.shape[0]), int(split[2] * data.shape[0])
    val_size = len(data)-train_size - test_size
    scaler.fit(data[:-test_size])  # ensure to scale values excluding the test set
    data_s = scaler.transform(data)
    if univariate=='True':
        data_s=data_s[:,-1:] #make it univariate
    else: 
        pass
    X_train,y_train=create_dataset(np.swapaxes(data_s, 1, 0)[:,:train_size], lookback, horizon)
    X_test,y_test=create_dataset(np.swapaxes(data_s, 1, 0)[:,-test_size-lookback:], lookback, horizon)
    return X_train.permute(0,2,1),X_test.permute(0,2,1),y_train.permute(0,2,1),y_test.permute(0,2,1)

class Configuration():
    def __init__(self,seq_len,pred_len):
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.enc_in=1
        self.e_layers=3
        self.n_heads=4
        self.d_model=16
        self.d_ff=128
        self.dropout=0.3
        self.fc_dropout=0.3
        self.head_dropout=0
        self.patch_len=24
        self.stride=2
        self.individual=0
        self.padding_patch='end'
        self.revin=1
        self.affine=0
        self.subtract_last=0
        self.decomposition=0
        self.kernel_size=25

def err_h(pred,y,variate='all'):
    '''Error horizon'''
    error_mse=[]
    error_mae=[]
    if variate !='all':
        v=variate
        for i in range(pred.shape[1]):
            error_mse.append(F.mse_loss(pred[:,i,v],y[:,i,v]).item())
            error_mae.append(F.l1_loss(pred[:,i,v],y[:,i,v]).item())
    else:
        for i in range(pred.shape[1]):
            error_mse.append(F.mse_loss(pred[:,i],y[:,i]).item())
            error_mae.append(F.l1_loss(pred[:,i],y[:,i]).item())

    return error_mse,error_mae

def err_f(pred,y,variate='all'):
    error_mse=[]
    error_mae=[]
    if variate!='all':
        v=variate
        for i in range(pred.shape[0]):
            error_mse.append(F.mse_loss(pred[i,:,v],y[i,:,v]).item())
            error_mae.append(F.l1_loss(pred[i,:,v],y[i,:,v]).item())
    else:
        for i in range(pred.shape[0]):
            error_mse.append(F.mse_loss(pred[i],y[i]).item())
            error_mae.append(F.l1_loss(pred[i],y[i,]).item())

    return error_mse,error_mae

def naive_predictor(X,horizon):
    '''Naive predictor'''
    pred=torch.zeros((X.shape[0],horizon,1))
    pred[:,:,:]=X[:,-1:,:]
    return pred


def random_subset(tensor, num_elements):
    """
    Selects a random subset from the given tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    num_elements (int): The number of elements in the subset.

    Returns:
    torch.Tensor: A tensor containing the random subset.
    """
    # Check if the number of elements requested is more than the number of elements in the tensor
    if num_elements > tensor.size(0):
        raise ValueError("Number of elements requested exceeds the number of elements in the tensor.")
    
    # Generate random indices
    random_indices = torch.randperm(tensor.size(0))[:num_elements]
    
    # Select the elements using the random indices
    subset = tensor[random_indices]
    
    return subset

def w_diff(classes,w_matrix):
    return w_matrix[:,:,classes[0]]-w_matrix[:,:,classes[1]]

def linear_xai(X,s):
    '''
    X: time series windows. [B,LxC]
    s: shap values. [B,LxC,h]
    '''
    X=X.unsqueeze(dim=-1)
    print(X.shape,s.shape)
    # s=torch.tensor(s)
    s_x=s/X
    s_x[torch.isinf(s_x)]=0
    xai_matrix=s_x[1:]-s_x[:-1]
    return s_x,torch.cat((torch.zeros_like(xai_matrix)[:1], xai_matrix), dim=0)

def train_model(model, dev, X_train, y_train, X_test, y_test, config):
    learning_rate = config["learning_rate"]
    loss_fn = nn.MSELoss()
    train_loader = data.DataLoader(data.TensorDataset(X_train[:int(X_train.shape[0]*0.8)], y_train[:int(y_train.shape[0]*0.8)].detach()), shuffle=True, batch_size=config['batch_size'])
    val_loader=data.DataLoader(data.TensorDataset(X_train[int(X_train.shape[0]*0.8):], y_train[int(y_train.shape[0]*0.8):].detach()), shuffle=False, batch_size=config['batch_size'])
    test_loader=data.DataLoader(data.TensorDataset(X_test, y_test.detach()), shuffle=False, batch_size=config['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = config['patience_scheduler'],threshold=5e-3)
    last_lr=learning_rate
    min_loss=float("inf")
    for ep in range(config['epochs']):
        model.train()
        loss_train=0
        for i,(X,y) in enumerate(train_loader):
            X,y=X.to(dev),y.to(dev)
            loss = F.mse_loss(model(X), y)
            loss_train+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del X 
            del y
        loss_train/= len(train_loader)
        model.eval()  
        with torch.no_grad():
            val_loss=0
            for X, y in val_loader:
                X,y=X.to(dev),y.to(dev)
                val_pred = model(X)
                val_loss += F.mse_loss(val_pred, y)
                del X
                del y
        val_loss /= len(val_loader) 
        if val_loss<min_loss:
            print(f'new best validation loss: {val_loss}')
            best_state_dict=model.state_dict() ##### TO SAVE
            # torch.save(best_state_dict, f'./results/{mod}trained_models/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}.pt')
            min_loss=val_loss
        with torch.no_grad():
            test_loss=0
            for X, y in test_loader:
                X,y=X.to(dev),y.to(dev)
                test_pred = model(X)
                test_loss += F.mse_loss(test_pred, y)
                del X
                del y
        test_loss /= len(test_loader)  
        scheduler.step(loss_train)        
        print("Epoch %d: train loss %.6f, val loss %.6f,  test loss %.6f " % (ep, loss_train,val_loss, test_loss))
    model.to(dev)
    model.eval()
    model.load_state_dict(best_state_dict)
    predictions=model(X_test).squeeze().cpu().detach()
    # if len(predictions.shape)>2:
    #     np.savetxt(f'./results/{mod}predictions/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_predictions.txt', predictions.reshape(-1,predictions.shape[1]*predictions.shape[2]))
    # else:
    #     np.savetxt(f'./results/{mod}predictions/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_predictions.txt', predictions)
    return model, predictions, loss_train, val_loss, test_loss

def get_SHAP_values(model, dev, X_train, X_test, config):
    model.to(dev)
    def model_for_shap(X):
        X=torch.tensor(X).float().reshape((-1,config['seq_len'], config['enc_in']))
        loader = data.DataLoader(X,batch_size=config['shap_batch_size'])
        preds=torch.tensor([])
        with torch.no_grad():
            for X in loader:
                X=X.to(dev)
                pred = model(X).cpu()
                del X
                preds=torch.cat((preds,pred),dim=0)
        return preds.detach().reshape(-1,preds.shape[1]*preds.shape[2]).numpy()
    if X_train.shape[0]>400:
        size=400
    else:
        size=X_train.shape[0]
    
    train_array=X_train.reshape(X_train.shape[0],config['seq_len']*config['enc_in']).cpu().detach().numpy()[np.random.choice(X_train.shape[0], size=size, replace=False)]
    test_array=X_test.reshape(X_test.shape[0],config['seq_len']*config['enc_in']).cpu().detach().numpy()
    # print(X_test.shape)
    # print(test_array.shape)
    explainer = shap.KernelExplainer(model_for_shap,train_array)
    shap_values = explainer(test_array)
    return shap_values

def get_close_idxs(indice, prev_size, prev_indices, j, sign):
    if j<0:
        return []
    elif j>=prev_size:
        return []
    elif abs(indice-prev_indices[j]) < 70 :
        return [prev_indices[j]] + get_close_idxs(indice, prev_size, prev_indices, j+sign, sign)
    else:
        return []

dataset_dict={'users':load_users,'euma':load_euma, 'pems03':load_PEMS03,'ili':load_basic,
'etth2':load_basic,'Vtraffic':load_basic,'synthetic':load_basic,
'earthquakes':load_UCR,'italypd':load_UCR,'chinatown':load_UCR,'plane':load_UCR,'yoga':load_UCR,'eumaclf':load_UCR,
'computers':load_UCR} #add here your loading function for your dataset

model_dict={'lstm': LSTM.Model,'patchtst': PatchTST.Model,'dlinear': DLinear.Model,
'neuroflex':NeuroFlexMLP.Model,'linear':Linear.Model,'tsmixer':TSMixer.Model,'linear_c':Linear_c.Model,
'tsmixer_c':TSMixer_c.Model,'patchtst_c': PatchTST_c.Model,'multirocket':MultiRocket.Model} 