import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Plot the figure used in the main paper from the csv file.
"""
sample_num = 50
directory = "figures/"
terminations = ["exhaustion"]
datasets = ["pub1", "pub2", "pub5"]
methods = ["gd", "ew"]

for termination in terminations:
  for dataset in datasets:
    for method in methods:
      df = pd.read_csv("results"+str(sample_num)+"_"+str(sample_num)+"_"+termination+"_"+dataset+method+".csv")
      plt.figure()
      plt.plot((df["T"]), df["Regret"], 'b-o')
      # plt.plot((df["T"]), df["Regret"], 'r-o')
      plt.xlabel("T", fontsize=16)
      plt.ylabel("Regret", fontsize=16)
      plt.fill_between(df["T"], df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5)
      plt.savefig(directory+str(sample_num)+"_"+termination+"_"+dataset+"_"+method+"_"+"regret.png")
      plt.close()

      # plt.figure()
      # plt.plot(np.sqrt(df["T"]), df["Regret"], '-o')
      # plt.xlabel("$\sqrt{T}$", fontsize=16)
      # plt.ylabel("Regret", fontsize=16)
      # plt.fill_between(np.sqrt(df["T"]), df["Regret"]-df["std"]/sample_num*3, df["Regret"]+df["std"]/sample_num*3, alpha=0.5)
      # plt.savefig(directory+str(sample_num)+"_"+termination+"_"+dataset+"_"+method+"_"+"_regret_sqrt.png")
      # plt.close()

      plt.figure()
      plt.plot((df["T"]), 1-df["Regret"]/df["OPT"], '-o')
      plt.xlabel("T", fontsize=16)
      plt.ylabel("Relative Regret", fontsize=16)
      plt.fill_between(df["T"], 1-(df["Regret"]+df["std"]/sample_num*3)/df["OPT"] , 1-(df["Regret"]-df["std"]/sample_num*3)/df["OPT"] , alpha=0.5)
      plt.savefig(directory+str(sample_num)+"_"+termination+"_"+dataset+"_"+method+"_"+"rel_regret.png")
      plt.close()