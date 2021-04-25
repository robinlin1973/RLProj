import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':

    sns.set_style("darkgrid")

    reward_records = np.load('log/dqn_reward.np', allow_pickle=True)
    dqn = np.var(reward_records, axis=1)
    normalized_dqn = dqn/np.linalg.norm(dqn)
    sns.lineplot( data=normalized_dqn,legend='brief', label='DQN')

    reward_records = np.load('log/ppo_reward.np', allow_pickle=True)
    ppo = np.var(reward_records, axis=1)
    normalized_ppo = ppo/np.linalg.norm(ppo)
    sns.lineplot( data=normalized_ppo,legend='brief', label='PPO')

    reward_records = np.load('log/ddpg_reward.np', allow_pickle=True)
    ddpg = np.var(reward_records, axis=1)
    normalized_ddpg = ddpg/np.linalg.norm(ddpg)
    sns.lineplot( data=normalized_ddpg,legend='brief', label='DDPG')

    # reward_records = np.load('log/ppod_reward.np', allow_pickle=True)
    # ppod = np.var(reward_records, axis=1)
    # normalized_ppod = ppod/np.linalg.norm(ppod)
    # sns.lineplot( data=normalized_ppod,legend='brief', label='PPOD')

    plt.savefig("img/vars.png")
    plt.show()
