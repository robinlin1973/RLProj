from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns

x_pxl, y_pxl = 400, 400
fig = plt.figure(figsize=(8, 20))
fig.suptitle('HEATMAP')
sns.color_palette("crest", as_cmap=True)

def format(ax,map,algo_name=None,title=None, cmap=plt.cm.summer):
    # im = ax.imshow(value_map)#, cmap=plt.cm.spring, interpolation='bicubic'
    g = sns.heatmap(map,cmap=cmap,ax=ax)
    if(title):
        g.set_title(title)

    g.set_xlabel('$\\theta$')
    g.set_xticks(np.linspace(0, x_pxl, 5))
    g.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    if(algo_name):
        g.set_ylabel(algo_name)

    g.set_yticks(np.linspace(0, y_pxl, 5))
    g.set_yticklabels(['-8', '-4', '0', '4', '8'])

def dqn_heatmap():
    from dqn import Net

    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    net = Net()
    net.load_state_dict(torch.load('param/dqn_net_params.pkl'))
    q = net(state)
    value_map = q.max(1)[0].view(y_pxl, x_pxl).detach().numpy()
    action_map = q.max(1)[1].view(y_pxl, x_pxl).detach().numpy() / 10 * 4 - 2

    format(plt.subplot(421),value_map, algo_name= 'DQN', title ="Value Map")
    # sns.heatmap(value_map, cmap="YlGnBu", linewidths=0.5, ax=plt.subplot(421))
    format(plt.subplot(422),action_map, algo_name= 'DQN', title ="Action Map",cmap= plt.cm.winter)

def ddpg_heatmap():
    from ddpg import ActorNet, CriticNet
    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    anet = ActorNet()
    anet.load_state_dict(torch.load('param/ddpg_anet_params.pkl'))
    action_map = anet(state).view(y_pxl, x_pxl).detach().numpy()

    cnet = CriticNet()
    cnet.load_state_dict(torch.load('param/ddpg_cnet_params.pkl'))
    value_map = cnet(state, anet(state)).view(y_pxl, x_pxl).detach().numpy()

    format(plt.subplot(423), value_map, algo_name= 'DDPQ')
    format(plt.subplot(424), action_map,  algo_name= 'DDPQ',cmap= plt.cm.winter)


def ppo_heatmap():
    from ppo import ActorNet, CriticNet
    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])
    cnet = CriticNet()
    cnet.load_state_dict(torch.load('param/ppo_cnet_params.pkl'))
    value_map = cnet(state).view(y_pxl, x_pxl).detach().numpy()

    anet = ActorNet()
    anet.load_state_dict(torch.load('param/ppo_anet_params.pkl'))
    action_map = anet(state)[0].view(y_pxl, x_pxl).detach().numpy()

    format(plt.subplot(425), value_map, algo_name= 'PPO')
    format(plt.subplot(426), action_map, algo_name= 'PPO',cmap= plt.cm.winter)


def ppo_d_heatmap():
    from ppo_d import ActorNet, CriticNet
    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    anet = ActorNet()
    anet.load_state_dict(torch.load('param/ppo_d_anet_params.pkl'))
    action_map = anet(state).max(1)[1].view(y_pxl, x_pxl).detach().numpy() / 10 * 4 - 2

    cnet = CriticNet()
    cnet.load_state_dict(torch.load('param/ppo_d_cnet_params.pkl'))
    value_map = cnet(state).view(y_pxl, x_pxl).detach().numpy()
    format(plt.subplot(427),value_map, algo_name= 'PPO_D')
    format(plt.subplot(428),action_map,algo_name= 'PPO_D',cmap= plt.cm.winter)

def main():
    dqn_heatmap()
    ddpg_heatmap()
    ppo_heatmap()
    ppo_d_heatmap()
    plt.tight_layout()
    plt.subplots_adjust( bottom=.1,top=0.9,hspace = .5)
    plt.savefig('img/heatmap.png')
    plt.show()


if __name__ == '__main__':
    main()
