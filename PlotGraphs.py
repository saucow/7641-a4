import matplotlib as mpl
mpl.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plotGraph(model="Easy", type="Policy"):
    file = model + type
    df = pd.read_csv(model+type + "-0_99.csv")
    df.set_index("iter")
    plotColumn(df, column_name="time", yLabel="Time", type=file)
    plotColumn(df, column_name="reward", yLabel="Reward", type=file)
    plotColumn(df, column_name="steps", yLabel="Steps", type=file)
    plotColumn(df, column_name="convergence", yLabel="Convergence", type=file)

def plotColumn(df, column_name, yLabel, type, xLabel="Iterations"):
    df[column_name].plot()
    # plt.xticks(df.index)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + " vs " + xLabel)
    plt.savefig(type + "-" + yLabel)
    # plt.show()
    plt.close()

def getQLearningFile(type="Easy",alpha=0.1, epsilon=0.1):
    alpha = str(alpha)
    alpha = alpha.replace(".", "_")
    epsilon = str(epsilon)
    epsilon = epsilon.replace(".", "_")
    return type+"Q-Learning_L"+alpha+"_E"+epsilon+"-0_99.csv"

def plotQLearning(type= "Easy", params = [(0.1, 0.1),  (0.1, 0.5),
                                          (0.3, 0.1),  (0.3, 0.5),
                                          (0.5, 0.1),  (0.5, 0.5),
                                          (0.9, 0.1), (0.9, 0.5)]):

    for column_name in ["time" , "reward", "steps", "convergence"]:
        legend = []
        for p in params:
            epsilon = p[1]
            alpha = p[0]
            f = getQLearningFile(type=type, alpha=alpha, epsilon=epsilon)
            df = pd.read_csv(f)
            df = df.set_index("iter")
            r = range(1, 10000, 250)
            r += [10000-1]
            df = df.iloc[r, :]
            l = "(alpha "+str(alpha) + ", epsilon " + str(epsilon) + ")"
            legend += [l]
            df[column_name].plot()

        plt.xlabel("Iterations")
        plt.ylabel(column_name)
        plt.legend(legend)
        plt.title(column_name + " vs " + "Iterations")
        plt.savefig(type + "Q-" + column_name)
        plt.close()

def plotDiscountRates(model="Easy", type="Policy"):

    discount_rates = [0.99, 0.89, 0.79, 0.69, 0.59]

    for column_name in ["time", "reward", "steps", "convergence"]:
        for r in discount_rates:
            r = str(r)
            r = r.replace(".", "_")
            df = pd.read_csv(model+type+"-"+r+".csv")
            df = df.set_index("iter")
            df[column_name].plot()

        plt.xlabel("Iterations")
        plt.ylabel(column_name.title())
        plt.legend(discount_rates)
        plt.title(column_name.title() + " vs " + "Iterations")
        plt.savefig(model+type+"-dr-"+ column_name)
        plt.close()

# plotGraph(model="Easy", type="Policy")
# plotGraph(model="Easy", type="Value")
# plotGraph(model="Hard", type="Policy")
# plotGraph(model="Hard", type="Value")

# plotDiscountRates(model="Easy", type="Policy")
# plotDiscountRates(model="Easy", type="Value")
# plotDiscountRates(model="Hard", type="Policy")
# plotDiscountRates(model="Hard", type="Value")
#
# plotQLearning(type="Easy")
plotQLearning(type="Hard")




