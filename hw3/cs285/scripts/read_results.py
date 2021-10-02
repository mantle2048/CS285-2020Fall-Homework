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
    steps = []
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


if __name__ == '__main__':
    plot_q1_lander_figure()


