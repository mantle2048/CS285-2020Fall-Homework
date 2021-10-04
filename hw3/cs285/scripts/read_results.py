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
    rewards = []
    best_rewards = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                rewards.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                best_rewards.append(v.simple_value)

    itor = len(steps)
    rewards = [rewards[0]] * (itor - len(rewards)) + rewards
    best_rewards = [rewards[0]] * (itor - len(best_rewards)) + best_rewards
    result_dict = dict(Train_steps=steps, Return=rewards, Best_Return=best_rewards, Iteration=list(range(0,itor)))# exp_name=[exp_name]*itor, env_name=[env_name]*itor)
    df = pd.DataFrame(result_dict)
    return df

def plot_data(data, xaxis='Train_steps', value='Return', condition='exp_name', smooth=1, ax= None):
    global imgdir
    if smooth > 1:
        for datum in data:
            datum[value]=uniform_filter1d(datum[value], size=smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    env_name = data['env_name'][0]
    sns.set(style='whitegrid', palette = 'tab10', font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, ci='sd', hue=condition)
    plt.legend(loc='best').set_draggable(True)
    plt.title(env_name)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)


def plot_q1_lander_figure():

    logdir = os.path.join(cur_dir, 'run_logs/hw3_q1_lander_LunarLander*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        return_data = event_data[['Train_steps', 'Return','Iteration']]
        return_data['exp_name'] = ['Train_AverageReturn'] * return_data.shape[0]
        best_return_data = event_data[['Train_steps', 'Best_Return','Iteration']]
        best_return_data['exp_name'] = ['Train_BestReturn'] * best_return_data.shape[0]
        best_return_data.columns = return_data.columns.values.tolist()
        event_data = return_data.append(best_return_data, ignore_index=True)
        print(event_data)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    plot_data(data,smooth=1, ax=ax)
    plt.savefig(os.path.join(imgdir, exp_name), dpi=300)

    print('==='*16)
    print('q1_lander figure was saved in folder image')
    print('==='*16)


def plot_q1_figure():

    logdir = os.path.join(cur_dir, 'run_logs/hw3_q1_MsPacman*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        return_data = event_data[['Train_steps', 'Return','Iteration']]
        return_data['exp_name'] = ['Train_AverageReturn'] * return_data.shape[0]
        best_return_data = event_data[['Train_steps', 'Best_Return','Iteration']]
        best_return_data['exp_name'] = ['Train_BestReturn'] * best_return_data.shape[0]
        best_return_data.columns = return_data.columns.values.tolist()
        event_data = return_data.append(best_return_data, ignore_index=True)
        print(event_data)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.set(xlim=(0, 1500001))
    plot_data(data,smooth=1, ax=ax)
    plt.savefig(os.path.join(imgdir, exp_name), dpi=300)

    print('==='*16)
    print('q1 figure was saved in folder image')
    print('==='*16)


def plot_q2_figure():

    logdir = os.path.join(cur_dir, 'run_logs/hw3_q2*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-2])
        exp_name_seed = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['exp_name_seed'] = [exp_name_seed] * event_data.shape[0]
        data.append(event_data)
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    # ax.set(xlim=(0, 1500001))
    plot_data(data,smooth=11, ax=ax, condition='exp_name')
    plt.savefig(os.path.join(imgdir, exp_name), dpi=300)

    print('==='*16)
    print('q2 figure was saved in folder image')
    print('==='*16)


def plot_q3_figure():

    my_cmp = lambda x:int(x.split('/')[-2].split('_')[-2])
    logdir = os.path.join(cur_dir, 'run_logs/hw3_q3*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        data.append(event_data)
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    plot_data(data,smooth=1, ax=ax, condition='exp_name')
    plt.savefig(os.path.join(imgdir, 'q3.png'), dpi=300)

    print('==='*16)
    print('q3 figure was saved in folder image')
    print('==='*16)


def plot_q4_figure():

    my_cmp = lambda x:(int(x.split('/')[-2].split('_')[-3]), int(x.split('/')[-2].split('_')[-2]))
    logdir = os.path.join(cur_dir, 'run_logs/hw3_q4*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        data.append(event_data)
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    plot_data(data,smooth=1, ax=ax, condition='exp_name')
    plt.savefig(os.path.join(imgdir, 'q4.png'), dpi=300)

    print('==='*16)
    print('q4 figure was saved in folder image')
    print('==='*16)


def plot_q5_figure():

    logdir = os.path.join(cur_dir, 'run_logs/hw3_q5*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir)):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[1:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['env_name'] = [env_name] * event_data.shape[0]
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        data.append(event_data)
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        plot_data(data,xaxis='Iteration', smooth=1, ax=ax, condition='exp_name')
        plt.savefig(os.path.join(imgdir, f'{exp_name}_{env_name}.png'), dpi=300)
        data = []

    print('==='*16)
    print('q5 figures were saved in folder image')
    print('==='*16)


if __name__ == '__main__':
    # plot_q1_lander_figure()
    # plot_q1_figure()
    # plot_q2_figure()
    # plot_q3_figure()
    # plot_q4_figure()
    plot_q5_figure()



