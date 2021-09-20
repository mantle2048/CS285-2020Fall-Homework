# %matplotlib notebook
import os
import matplotlib.pyplot as plt
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
from collections import defaultdict

# +
cur_dir = os.getcwd()
if 'image' in cur_dir.split('/'):
    data_path = os.path.join(cur_dir, 'hyperparameters/')
    dagger_data_path = os.path.join(cur_dir, 'dagger/Humanoid')
    fig_save_path = os.path.join(cur_dir, 'figure/')
else:
    data_path = os.path.join(cur_dir, 'image/hyperparameters/')
    dagger_data_path = os.path.join(cur_dir, 'image/dagger/Humanoid')
    fig_save_path = os.path.join(cur_dir, 'image/figure/')
    
my_cmp = lambda x: int(x.split('_')[3])


# -

def tabulate_events(data_path):
    data_dirs =  sorted(os.listdir(data_path), key=my_cmp)
    summary_iterators = [EventAccumulator(os.path.join(data_path, data_name)).Reload() for data_name in data_dirs]
    
    tags = summary_iterators[0].Tags()['scalars']
    
    for itor in summary_iterators:
        assert itor.Tags()['scalars'] == tags
    tabulate_datas = defaultdict(list)
    training_steps = [data_dir.split('_')[3] for data_dir in data_dirs]
    n_iters = [data_dir.split('_')[5] for data_dir in data_dirs]
    env_names = [data_dir.split('_')[6] for data_dir in data_dirs]
    tabulate_datas['Training_steps'] = training_steps
    tabulate_datas['n_iter'] = n_iters
    tabulate_datas['env_name'] = env_names
    steps = []
    
    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        
        for events in zip(*[summary.Scalars(tag) for summary in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            tabulate_datas[tag].extend([e.value for e in events])
    return tabulate_datas, steps


def get_dataframe(data_path):
    dirs = sorted(os.listdir(data_path), key=my_cmp)
    train_steps_per_iter = [data_dir.split('_')[3] for data_dir in dirs]
    tabulate_datas, steps = tabulate_events(data_path)
    df = pd.DataFrame(tabulate_datas)
    return df
get_dataframe(data_path)


# +
def plot_data(data, xaxis='Training_steps',value='AverageReturn', std='StdReturn', ax=None, **kwargs):
    env_name = data['env_name'][0]
    x = np.array(data[xaxis])
    eval_return = np.array(data['Eval_'+value])
    eval_std = np.array(data['Eval_'+std])
    
    train_return = np.array(data['Train_'+value])
    train_std = np.array(data['Train_'+std])
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    plt.errorbar(x, eval_return, eval_std, linestyle='--', elinewidth=2, marker='o',capsize=8,capthick=1, label='Behavior Cloning')
    plt.errorbar(x, train_return, train_std,linestyle='--',elinewidth=2, marker='o',capsize=8,capthick=1, label='Expert')
    plt.xlabel('Training steps')
    plt.ylabel('AverageReturn')
    plt.title(env_name)
    plt.legend()
    plt.show()
    
    plt.savefig(os.path.join(fig_save_path,f'bc_{env_name}.png'), dpi=300)
    print(f'Figure was saved in {fig_save_path}')
    

# -

def tabulate_dagger_events(data_path):
    data_dirs =  sorted(os.listdir(data_path), key=my_cmp)
    summary_iterators = [EventAccumulator(os.path.join(data_path, data_name)).Reload() for data_name in data_dirs]
    
    tags = summary_iterators[0].Tags()['scalars']
    
    for itor in summary_iterators:
        assert itor.Tags()['scalars'] == tags
    tabulate_datas = defaultdict(list)
    env_name = data_dirs[0].split('_')[6]
    steps = []
    
    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        
        for events in zip(*[summary.Scalars(tag) for summary in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            tabulate_datas[tag].extend([e.value for e in events])
            
    tabulate_datas['Steps'] = steps
    
    BC_AverageReturn = [tabulate_datas['Eval_AverageReturn'][0] for _ in range(len(steps))]
    BC_StdReturn = [tabulate_datas['Eval_StdReturn'][0] for _ in range(len(steps))]
    
    Expert_AverageReturn = [tabulate_datas['Train_AverageReturn'][0] for _ in range(len(steps))]
    Expert_StdReturn = [tabulate_datas['Train_StdReturn'][0] for _ in range(len(steps))]
    
    tabulate_datas['BC_AverageReturn'] = BC_AverageReturn
    tabulate_datas['BC_StdReturn'] = BC_StdReturn
    
    tabulate_datas['Expert_AverageReturn'] = Expert_AverageReturn
    tabulate_datas['Expert_StdReturn'] = Expert_StdReturn
    
    
    tabulate_datas['env_name'] = np.array([env_name for _ in range(len(steps))])
    
    return tabulate_datas, steps


# +
def get_dagger_dataframe(data_path):
    dirs = sorted(os.listdir(data_path), key=my_cmp)
    train_steps_per_iter = [data_dir.split('_')[3] for data_dir in dirs]
    tabulate_datas, steps = tabulate_dagger_events(data_path)
    df = pd.DataFrame(tabulate_datas)
    return df

get_dagger_dataframe(dagger_data_path) 


# -

def plot_dagger_data(data, xaxis='Steps',value='AverageReturn', std='StdReturn', ax=None, **kwargs):
    env_name = data['env_name'][0]
    x = np.array(data[xaxis])
    eval_return = np.array(data['Eval_'+value])
    eval_std = np.array(data['Eval_'+std])
    
    bc_return = np.array(data['BC_'+value])
    bc_std = np.array(data['BC_'+std])
    
    
    expert_return = np.array(data['Expert_'+value])
    expert_std = np.array(data['Expert_'+std])
    
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    plt.errorbar(x, eval_return, eval_std, linestyle='--', elinewidth=2, marker='o',capsize=8,capthick=1, label='DAgger')
    plt.errorbar(x, bc_return, bc_std,linestyle='--',elinewidth=2, marker='o',capsize=8,capthick=1, label='Behavior Cloning')
    plt.errorbar(x, expert_return, expert_std,linestyle='--',elinewidth=2, marker='o',capsize=8,capthick=1, label='Expert')
    plt.xlabel('n_iter')
    plt.ylabel('AverageReturn')
    plt.ylim([0, 11000])
    plt.title(env_name)
    plt.legend()
    plt.show()
    
    image_name = os.path.join(fig_save_path,f'dagger_{env_name}.png')
    plt.savefig(image_name, dpi=300)
    print(f'Figure {image_name} was saved in {fig_save_path}')
if __name__ == '__main__':
    data = get_dataframe(data_path)
    plot_data(data)
    dagger_data =  get_dagger_dataframe(dagger_data_path)
    plot_dagger_data(dagger_data)


