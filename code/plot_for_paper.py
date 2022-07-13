import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# from cycler import cycler
# plt.style.use('ggplot')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

sample_num = 50
directory = "figures/"
terminations = ["exhaustion"]
datasets = ["pub1", "pub2", "pub5"]
methods = ["gd", "ew"]
colors = ["red", "blue", "black", "green"]
titles = ["Online Gradient Descent", "Multiplicative Weight Updates"]
# ylables = ["Regret", "Relative Reward"]

# # Plot Regret
# for termination in terminations:
#   for j, method in enumerate(methods):
#     plt.figure()
#     for i, dataset in enumerate(datasets):
#       df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+"True.csv")
#       plt.plot((df["T"]), df["Regret"], '-o', color=colors[i], label=dataset)
#       plt.fill_between(df["T"], df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5, color=colors[i])
#     plt.legend(loc="upper left")
#     plt.title(titles[j], fontsize=20)
#     plt.xlabel("T")
#     plt.ylabel("Regret")
#     plt.savefig(directory+method+"_"+"regret.png", dpi=200)
#     plt.close()

# # Plot Relative Reward
# for termination in terminations:
#   for j, method in enumerate(methods):
#     plt.figure()
#     for i, dataset in enumerate(datasets):
#       df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+"True.csv")
#       plt.plot((df["T"]), 1-df["Regret"]/df["OPT"], '-o', color=colors[i], label=dataset)
#       # plt.fill_between(df["T"], df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5, color=colors[i])
#     plt.legend(loc="upper left")
#     plt.title(titles[j], fontsize=20)
#     plt.xlabel("T")
#     plt.ylabel("Relative Reward")
#     plt.savefig(directory+method+"_"+"relative_reward.png", dpi=200)
#     plt.close()

# # Plot lambda
# lambds = ["002", "0002", "00002"]
# lambd_values = ["0.02", "0.002", "0.0002"]
# dataset = "pub2"
# ss = "1"
# method = "gd"
# for termination in terminations:
#   for i, lambd in enumerate(lambds):
#     plt.figure()
#     df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+"lambda"+lambd+"ss"+ss+"True.csv")
#     plt.plot((df["T"]), df["Real_Reward"], '-o', color=colors[0], label="Real Reward")
#     plt.plot((df["T"]), df["Reward"]-df["Real_Reward"], '--o', color=colors[1], label="Regularization")
#     # plt.fill_between(df["T"], df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5, color=colors[i])
#     plt.legend(loc="upper left")
#     plt.title("$\lambda$="+lambd_values[i], fontsize=20)
#     plt.xlabel("T")
#     plt.ylabel("Reward")
#     plt.savefig(directory+method+"_"+"lambd"+lambds[i]+".png", dpi=200)
#     plt.close()

# # Plot eta
# lambd = "0002"
# dataset = "pub2"
# sss = ["01", "1", "10"]
# etas = ["0.1", "1", "10"]
# method = "gd"
# for termination in terminations:
#   plt.figure()
#   for i, ss in enumerate(sss):
#     df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+"lambda"+lambd+"ss"+ss+"True.csv")
#     plt.plot((df["T"]), df["Regret"], '-o', color=colors[i], label=r'$\alpha$='+etas[i])
#   plt.legend(loc="upper left")
#   plt.title("Regret versus step-size", fontsize=20)
#   plt.xlabel("T")
#   plt.ylabel("Regret")
#   plt.savefig(directory+method+"_"+"ss.png", dpi=200)
#   plt.close()

# Plot regret for different lambdas
lambds = ["01", "001", "0001", "00001", "00"]
lambd_values = [0.1, 0.01, 0.001, 0.0001, 0.0]
dataset = "pub2"
reference = "scaled_square_norm"
regularizer = "_maxmin_fairness"
ss = "001"
num_trials = "30"

plt.figure()
for i, lambd in enumerate(lambds):
  df = pd.read_csv("output/results"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  plt.plot((df["T"]), df["Regret"], '-o', label=r'$\lambda$='+str(lambd_values[i]))
  # color_inx += 1
  plt.legend(loc="upper left")
  # plt.title("Regret", fontsize=20)
  plt.xlabel("T", fontsize=20)
  plt.ylabel("Regret", fontsize=20)
plt.savefig("figure/regret_different_lambda"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
plt.close()

# Plot real for different lambdas
plt.figure()
for i, lambd in enumerate(lambds):
  df = pd.read_csv("output/results"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  plt.plot((df["T"]), df["Real_Reward"], '-o', label=r'$\lambda$='+str(lambd_values[i]))
  # color_inx += 1
  plt.legend(loc="upper left")
  plt.title("Real Reward", fontsize=20)
  plt.xlabel("T")
  plt.ylabel("Real Reward")
plt.savefig("figure/real_reward_different_lambda"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
plt.close()

# Plot SI
num_col = 12
for i, lambd in enumerate(lambds):
  plt.figure()
  matrix = np.loadtxt("output/si"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  # print(matrix)
  # print(lambd)
  for j in range(num_col):
    plt.plot(np.linspace(0, 9999, 100)[1:], matrix[1:,j])
  plt.title("SI with " + r'$\lambda$='+str(lambd_values[i]))
  plt.xlabel("T")
  plt.ylabel("SI")
  plt.savefig("figure/si_lambda"+lambd+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
  # plt.show()
  plt.close()

# Plot relative reward for different lambdas
plt.figure()
for i, lambd in enumerate(lambds):
  df = pd.read_csv("output/results"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  plt.plot((df["T"]), 1-df["Regret"]/df["OPT"], '-o', label=r'$\lambda$='+str(lambd_values[i]))
  # color_inx += 1
  plt.legend(loc="lower right")
  plt.title("Relative Regularized Reward", fontsize=20)
  plt.xlabel("T")
  plt.ylabel("Relative Regularized Reward")
plt.savefig("figure/relative_reward_different_lambda"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
plt.close()

# Plot reward-regularization figure
fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
dfs = []
for i, lambd in enumerate(lambds[:]):
  df = pd.read_csv("output/results"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  dfs.append(df)

scaled_regularizer = np.array([])
scaled_real_reward = np.array([])
T = dfs[i].at[10, "T"]
j = 10
for i, lambd in enumerate(lambds[:]):
  print dfs[i]
  # scaled_regularizer = np.append(scaled_regularizer, (dfs[i].at[j, "Reward"]-dfs[i].at[j, "Real_Reward"])/T/lambd_values[i])
  # scaled_regularizer = np.append(scaled_regularizer, (dfs[i].at[j, "Reward"]-dfs[i].at[j, "Real_Reward"])/T/lambd_values[i])
  scaled_regularizer = np.append(scaled_regularizer, dfs[i].at[j, "Regularization"])
  scaled_real_reward = np.append(scaled_real_reward, (dfs[i].at[j, "Real_Reward"]))

# print(lambd_values[:3], scaled_regularizer, scaled_real_reward)
plt.plot(scaled_regularizer, scaled_real_reward, "-o", label="average revenue")
# plt.plot(scaled_regularizer, label="T="+str(T))
# plt.legend()
# plt.xscale("log")
# ax.set_xlabel('Minimal Consumption Ratio')
# ax.set_ylabel('Total Revenue')
plt.xlim(0.3,0.68)
plt.ylim(140,155)
plt.xlabel("Max-Min Fairness", fontsize=20)
plt.ylabel("Reward", fontsize=20)
# plt.show()
plt.savefig("figure/reward-regularization"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
# plt.close()


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Plot ellipsoids
# fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i, lambd in enumerate(lambds[:]):
  scaled_regularizer = np.loadtxt("output/regularization"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  scaled_real_reward = np.loadtxt("output/real_revenue"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5_num_trials_"+num_trials+".csv")
  confidence_ellipse(scaled_regularizer, scaled_real_reward, ax, n_std=2/np.sqrt(30))
plt.xlim(0.25,0.7)
plt.ylim(140,160)
plt.show()
plt.savefig("figure/reward-regularization-ellipsoid"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)


# # Plot reward-regularization figure
# plt.figure()
# dfs = []
# for i, lambd in enumerate(lambds[:3]):
#   df = pd.read_csv("output/results"+"_"+dataset+"_"+"lambda"+lambd+"_ss"+ss+regularizer+"_"+reference+"_sumrho_1.5.csv")
#   dfs.append(df)
# for j in [0, 1, 10]:
#   scaled_regularizer = np.array([])
#   scaled_real_reward = np.array([])
#   T = dfs[i].at[0, "T"]
#   for i, lambd in enumerate(lambds[:3]):
#     print dfs[i]
#     # scaled_regularizer = np.append(scaled_regularizer, (dfs[i].at[j, "Reward"]-dfs[i].at[j, "Real_Reward"])/T/lambd_values[i])
#     scaled_regularizer = np.append(scaled_regularizer, lambd_values[i])
#     scaled_real_reward = np.append(scaled_real_reward, (dfs[i].at[j, "Real_Reward"])/T)
#   print(scaled_regularizer, scaled_real_reward)
#   plt.plot(scaled_real_reward, scaled_regularizer, label="T="+str(T))
# plt.xscale("log")
# plt.show()
  
    # print(dfs[i].at[j, "T"])

# plt.plot((df["T"]), 1-df["Regret"]/df["OPT"], '-o', label=r'$\lambda$='+str(lambd_values[i]))
# # color_inx += 1
# plt.legend(loc="lower right")
# plt.title("Relative Regularized Reward", fontsize=20)
# plt.xlabel("T")
# plt.ylabel("Relative Regularized Reward")
# plt.savefig("figure/relative_reward_different_lambda"+"_ss"+ss+regularizer+"_"+"square_norm"+".png", dpi=200)
# plt.close()


# # Plot Relative Reward
# for termination in terminations:
#   for j, method in enumerate(methods):
#     plt.figure()
#     for i, dataset in enumerate(datasets):
#       df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+"True.csv")
#       plt.plot((df["T"]), 1-df["Regret"]/df["OPT"], '-o', color=colors[i], label=dataset)
#       # plt.fill_between(df["T"], df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5, color=colors[i])
#     plt.legend(loc="upper left")
#     plt.title(titles[j], fontsize=20)
#     plt.xlabel("T")
#     plt.ylabel("Relative Reward")
#     plt.savefig(directory+method+"_"+"relative_reward.png", dpi=200)
#     plt.close()


