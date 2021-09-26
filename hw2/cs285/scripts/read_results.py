import glob
import tensorflow as tf
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib notebook
import numpy as np
from scipy.ndimage import uniform_filter1d

# +
cur_dir = os.getcwd()
if 'scripts' in cur_dir.split('/'):
    cur_dir = os.path.abspath("../..")

imgdir = os.path.join(cur_dir, 'image')
os.makedirs(imgdir, exist_ok=True)


# -

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    steps = [0]
    rewards = [0]
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                rewards.append(v.simple_value)
    itor = len(steps)
    result_dict = dict(Train_steps=steps, Return=rewards, Iteration=list(range(0,itor)))# exp_name=[exp_name]*itor, env_name=[env_name]*itor)
    df = pd.DataFrame(result_dict)
    return df

def plot_data(data, xaxis='Iteration', value='Return', condition='exp_name', smooth=1):
    global imgdir
    if smooth > 1:
        for datum in data:
            datum[value]=uniform_filter1d(datum[value], size=smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    env_name = data['env_name'][0]
    sns.set(style='whitegrid', palette = 'tab10', font_scale=1.5)
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd')
    plt.legend(loc='best').set_draggable(True)
    plt.title(env_name)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)


def plot_q1_figure():
    
    logdir = os.path.join(cur_dir, 'run_logs/q1_sb*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[7].split('_')[0:4])
        env_name = eventfile.split('/')[7].split('_')[4]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    size = data
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:5]+'.png'), dpi=300)
    
    logdir = os.path.join(cur_dir, 'run_logs/q1_lb*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[7].split('_')[0:4])
        env_name = eventfile.split('/')[7].split('_')[4]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:5]+'.png'), dpi=300)
    
    print('==='*16)
    print('q1 figures were saved in folder image')
    print('==='*16)


def plot_q2_figure():
    my_cmp = lambda x: (int(x.split('/')[7].split('_')[2]), float(x.split('/')[7].split('_')[4]))
    logdir = os.path.join(cur_dir, 'run_logs/q2*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[7].split('_')[0:5])
        env_name = eventfile.split('/')[7].split('_')[5]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=3)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'.png'), dpi=300)
    

    print('==='*16)
    print('q2 figures were saved in folder image')
    print('==='*16)

if __name__ == '__main__':
    pass
    # plot_q1_figure()
    # plot_q2_figure()


