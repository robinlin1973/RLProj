from matplotlib import pyplot as plt
import numpy as np
import torch
import pickle
from collections import namedtuple
import seaborn as sns

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0, 700, 100))
ax.set_yticks(np.arange(-1500, 0, 200))
TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
def draw_line(nwk):
    with open('log/'+nwk+'_training_records.pkl', 'rb') as f:
        training_log = pickle.load(f)

    plt.plot([r.ep for r in training_log], [r.reward for r in training_log],label=nwk)
    # sns.lineplot(data=training_log, x="ep", y="reward")

def main():
    draw_line('dqn')
    draw_line('ddpg')
    draw_line('ppo')
    draw_line('ppo_d')


    plt.title('Algorithm plot')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig("img/lines.png")
    plt.show()

if __name__ == '__main__':
    main()
