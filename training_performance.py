# PLOTTING
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def plot():
    files = glob.glob('dqn_results/**/*.csv')
    dfs = [pd.read_csv(fp, skipinitialspace=True, usecols=['episode_reward_mean','timesteps_total']) for fp in files]
    df = pd.concat(dfs, axis=1)

    # Style Setting
    font_names = [f.name for f in fm.fontManager.ttflist]
    mpl.rcParams['font.family'] = 'Liberation Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 2
    mpl.style.use('seaborn-ticks')


    # Plotting
    x = df.iloc[:,1]
    df.drop(columns = ['timesteps_total'], inplace = True)
    y = df.mean(axis = 1)
    error = df.std(axis = 1)
    plt.plot(x,y)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward Mean')
    plt.fill_between(x, y+error, y-error, facecolor = 'blue', alpha = .5)
    plt.title("Training performance of DQN", y=-0.30)    
    plt.savefig("plots/dqn_training.svg", bbox_inches='tight')

    print('Saved Training Plot')

if __name__ == "__main__":
    plot()



