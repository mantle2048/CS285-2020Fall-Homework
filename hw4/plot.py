# +
# %matplotlib notebook
from typing import List, Dict

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
data_dir = os.path.join(os.getcwd(), 'run_logs')
img_dir = os.path.join(os.getcwd(), 'image')


# -

def get_event_data(event_dir: str) -> pd.DataFrame:
    exp_name = event_dir.split('/')[-3] 
    run_name = event_dir.split('/')[-2]
    EvalAverageReturn = []
    Steps = []
    
    for event in tf.train.summary_iterator(event_dir):
        for value in event.summary.value:
            if value.tag == 'Eval_AverageReturn':
                EvalAverageReturn.append(value.simple_value)
            elif value.tag == 'Train_EnvstepsSoFar':
                Steps.append(value.simple_value)
 
    assert len(EvalAverageReturn) == len(Steps), "Data log length must to equal"
    Iteration = np.arange(len(Steps))
    data_dict = dict(Steps=Steps, EvalAverageReturn=EvalAverageReturn, Iteration=Iteration, Condition1=exp_name, Condition2=run_name)
    data = pd.DataFrame(data_dict)
    return data


def get_exp_data(events_dir: str) -> List[pd.DataFrame]:
    exp_data = []
    for event_dir in events_dir:
        exp_data.append(get_event_data(event_dir))
    return exp_data


def plot_data(data: pd.DataFrame, ax=None, xaxis='Iteration', value='EvalAverageReturn', condition='Condition2', smooth=1):
    if smooth > 1:
        if isinstance(data, list):
            for datam in data:
                datam[value]=uniform_filter1d(data[value], size=smooth)
        else:
            data[value]=uniform_filter1d(data[value], size=smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    env_name = data['Condition1'][0].split('_')[-1]
    sns.set(style='whitegrid', palette='tab10', font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', ax=ax, linewidth=3.0)
    leg = ax.legend(loc='best') #.set_draggable(True)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    ax.set_title(env_name)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


def read_exp_data_and_plot(log_dir: str, exp_names: List, plot_config: Dict={}) -> List[pd.DataFrame]:
    exp_num = len(exp_names)
    fig, axs = plt.subplots(1, exp_num, figsize=(exp_num*8, 6))
    if not isinstance(axs, np.ndarray): axs = [axs]
    for ax, exp_name in zip(axs, exp_names):
        exp_dir = os.path.join(log_dir, f'*{exp_name}*')
        events_dir = glob.glob(os.path.join(exp_dir, '*', '*events*'))
        exp_data = get_exp_data(events_dir)
        plot_data(data=exp_data, ax=ax, **plot_config)
    plt.tight_layout(pad=0.5)


# +
def plot_prob3(prob_idx=3):
    
    global img_dir
    global data_dir
    exp_names = ['cheetah', 'obstacles', 'reacher']
    log_dir = os.path.join(data_dir, f'prob{prob_idx}')
    print(log_dir)
    read_exp_data_and_plot(log_dir, exp_names)
    plt.savefig(fname=os.path.join(img_dir, f'problem{prob_idx}.png'), dpi=300)
    
# plot_prob3()


# +
def plot_prob4(prob_idx=4):
    global img_dir
    global data_dir
    exp_names = ['ensemble', 'horizon', 'numseq']
    # exp_names = ['numseq']
    log_dir = os.path.join(data_dir, f'prob{prob_idx}')
    read_exp_data_and_plot(log_dir, exp_names)
    plt.savefig(fname=os.path.join(img_dir, f'problem{prob_idx}.png'), dpi=300)
    
# plot_prob4()


# +
def plot_prob5(prob_idx=5):
    global img_dir
    global data_dir
    exp_names = ['cheetah']
    log_dir = os.path.join(data_dir, f'prob{prob_idx}')
    plot_config = {'smooth':1, 'xaxis':'Iteration', 'condition':'Condition2'}
    read_exp_data_and_plot(log_dir, exp_names, plot_config)
    plt.savefig(fname=os.path.join(img_dir, f'problem{prob_idx}.png'), dpi=300)
    
plot_prob5()
# -




