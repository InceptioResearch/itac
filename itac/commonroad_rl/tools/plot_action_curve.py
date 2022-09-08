import matplotlib.pyplot as plt
import pickle

with open("continuous_actions.pkl", "rb") as f:
    continuous_actions = pickle.load(f)
with open("discrete_actions.pkl", "rb") as f:
    discrete_actions = pickle.load(f)
# continuous_actions = np.load("continuous_actions.npz", allow_pickle=True)
# discrete_actions = np.load("discrete_actions.npz", allow_pickle=True)

discrete_accelerations = discrete_actions["accelerations"]
discrete_jerks = discrete_actions["jerks"]
continuous_acceleraitons = continuous_actions["accelerations"]
continuous_jerks = continuous_actions["accelerations"]

for key in discrete_accelerations.keys():
    plt.figure()
    # accelerations
    plt.subplot(2, 2, 1)
    plt.title("accleration-x")
    plt.plot(discrete_accelerations[key][:, 0], color="blue", label="discrete")
    plt.plot(continuous_acceleraitons[key][:, 0], color="red", label="continuous")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("accleration-y")
    plt.plot(discrete_accelerations[key][:, 1], color="blue", label="discrete")
    plt.plot(continuous_acceleraitons[key][:, 1], color="red", label="continuous")
    # plt.legend()
    # jerks
    plt.subplot(2, 2, 3)
    plt.title("jerk-x")
    plt.plot(discrete_jerks[key][:, 0], color="blue", label="discrete")
    plt.plot(continuous_jerks[key][:, 0], color="red", label="continuous")
    # plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("jerk-y")
    plt.plot(discrete_jerks[key][:, 1], color="blue", label="discrete")
    plt.plot(continuous_jerks[key][:, 1], color="red", label="continuous")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{key}.png")