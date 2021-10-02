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
    fig, ax = plt.subplots(1,1,figsize=(12,8))
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
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:4])
        env_name = eventfile.split('/')[-2].split('_')[4]
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
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:4])
        env_name = eventfile.split('/')[-2].split('_')[4]
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
    my_cmp = lambda x: (int(x.split('/')[-2].split('_')[2]), float(x.split('/')[-2].split('_')[4]))
    logdir = os.path.join(cur_dir, 'run_logs/q2*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:5])
        env_name = eventfile.split('/')[-2].split('_')[5]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=3)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'.png'), dpi=300)
    

    print('==='*16)
    print('q2 figures were saved in folder image')
    print('==='*16)


def plot_q3_figure():
    logdir = os.path.join(cur_dir, 'run_logs/q3*/events*')
    data = []
    for eventfile in glob.glob(logdir):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:2])
        env_name = eventfile.split('/')[-2].split('_')[3]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'.png'), dpi=300)
    

    print('==='*16)
    print('q3 figures were saved in folder image')
    print('==='*16)


def plot_q4_1_figure():
    my_cmp = lambda x: (int(x.split('/')[-2].split('_')[3]), float(x.split('/')[-2].split('_')[5]))
    logdir = os.path.join(cur_dir, 'run_logs/q4_search*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:8])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'_1.png'), dpi=300)
    

    print('==='*16)
    print('q4_1 figures were saved in folder image')
    print('==='*16)


def plot_q4_2_figure():
    my_cmp = lambda x: len(x.split('/')[-2].split('_'))
    logdir = os.path.join(cur_dir, 'run_logs/q4_b*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:8])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'_2.png'), dpi=300)
    

    print('==='*16)
    print('q4_2 figures were saved in folder image')
    print('==='*16)


def plot_q5_figure():
    my_cmp = lambda x: float(x.split('/')[-2].split('_')[-2])
    logdir = os.path.join(cur_dir, 'run_logs/q5*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'.png'), dpi=300)
    

    print('==='*16)
    print('q5 figures were saved in folder image')
    print('==='*16)


def traintime_wrapper(fn):
    def inner(file):
        df = fn(file)
        train_times = []
        for e in tf.train.summary_iterator(file):
            for v in e.summary.value:
                if v.tag == 'TimeSinceStart':
                    train_times.append(v.simple_value)
        print('train_time:',train_times[-1])
        return  df
    return inner


def plot_q6_figure():
    my_cmp = lambda x: float(x.split('/')[-2].split('_')[-2])
    logdir = os.path.join(cur_dir, 'run_logs/q[36]*/events*')
    data = []
    get_section_results_with_train_time = traintime_wrapper(get_section_results)
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results_with_train_time(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=1)
    plt.savefig(os.path.join(imgdir, 'q6.png'), dpi=300)
    

    print('==='*16)
    print('q6 figures were saved in folder image')
    print('==='*16)


def plot_q7_figure():
    my_cmp = lambda x: float(x.split('/')[-2].split('_')[-4])
    logdir = os.path.join(cur_dir, 'run_logs/q7*/events*')
    data = []
    for eventfile in sorted(glob.glob(logdir), key=my_cmp):
        exp_name = '_'.join(eventfile.split('/')[-2].split('_')[0:-1])
        env_name = eventfile.split('/')[-2].split('_')[-1]
        event_data = get_section_results(eventfile)
        event_data['exp_name'] = [exp_name] * event_data.shape[0]
        event_data['env_name'] = [env_name] * event_data.shape[0]
        data.append(event_data)
    plot_data(data, smooth=7)
    plt.savefig(os.path.join(imgdir, data[0]['exp_name'][0][0:2]+'.png'), dpi=300)
    

    print('==='*16)
    print('q7 figures were saved in folder image')
    print('==='*16)

if __name__ == '__main__':
    # plot_q1_figure()
    # plot_q2_figure()
    # plot_q3_figure()
    # plot_q4_1_figure()
    # plot_q4_2_figure()
    # plot_q5_figure()
    plot_q6_figure()
    # plot_q7_figure()




