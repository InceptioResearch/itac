import argparse
import os
import re
import glob
import numpy as np
import yaml

import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

from SALib.sample import saltelli, fast_sampler, latin, morris, ff
from SALib.analyze import sobol, fast, rbd_fast, delta
from SALib.analyze import morris as morris_analyze
from SALib.analyze import ff as ff_analyze

from stable_baselines.common.vec_env import VecNormalize
from stable_baselines import PPO2
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from commonroad_rl.utils_run.utils import ALGOS, get_wrapper_class, make_env
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

from commonroad_rl.gym_commonroad.utils.scenario_io import get_project_root

# TODO DGSM
# TODO ff doesnt work due to padding --> workaround?
os.environ["KMP_WARNINGS"] = "off" 

SAMPLER = {
    "sobol": saltelli,
    "fast": fast_sampler,
    "rbd_fast": latin,
    "morris": morris,
    "delta": latin,
    "ff": ff
}

ANALYZERS = {
    "sobol": sobol,
    "fast": fast,
    "rbd_fast": rbd_fast,
    "morris": morris_analyze,
    "delta": delta,
    "ff": ff_analyze
}

RESULTS = {
    "sobol": ['S1', 'S2', 'ST'],
    "fast": ['S1', 'ST'],
    "rbd_fast": ['S1'],
    "morris": ['mu', 'mu_star', 'sigma', 'mu_star_conf'],
    'delta': ['delta', 'delta_conf', 'S1', 'S1_conf'],
    "ff": ['ME', 'IE'],
    }


def get_parser():
    """
    Initializes Parser
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", help="Set the number of Samples to be considered", default=int(256), type=int)
    parser.add_argument("--algo", type=str, default="ppo2")
    parser.add_argument("--method", type=str, default=None, help="Sensitivity Analysis methods: sobol, fast, rbd_fast, morris")
    parser.add_argument("--config_filename", "-config_f", type=str, default="environment_configurations.yml", 
                        help="Name of the configuration file, default: environement_configurations.yml")
    parser.add_argument("--model_path", "-model", type=str, help="Path to trained model",
                        default=PATH_PARAMS["log"] + "/ppo2/commonroad-v1_1")
    parser.add_argument("--save_path", type=str, help="Path where the data should be saved",
                        default=PATH_PARAMS["log"])
    parser.add_argument("--data_path", type=str, help="Path to pickle files",
                        default=PATH_PARAMS["pickles"])
    parser.add_argument("--save_fig", action="store_true", help="Store plots for analysis")
    parser.add_argument("--save_data", action="store_true", help="Store data for analysis as .npy files")
    parser.add_argument("--save_gif", action="store_true", help="Store data for analysis as gif")
    parser.add_argument("--n_frames", type=int, default=5)

    return parser

def args_assertions(args):
    """
    This function collects assertions related to the parsed arguments

    :param args: the arguments input through the parser
    """

    assert (args.n & (args.n-1) == 0) and args.n != 0, '-n given as ' + str(args.n) + ' needs to be of 2^(x) with x a natural Number'
    assert args.method in ANALYZERS, '--method given as ' + args.method + ' is not in possible methods: sobol, fast, rbd_fast, morris'
    assert args.algo in ALGOS, '--alorithn given as ' + args.method + ' is not in possible alorithms: a2c, acer, acktr, ddpg, her, sac, pp2, trpo, td3'


def open_configs_sens_bounds():
    """
    This function loads the bounds from the config.yaml file in PATH_PARAMS["configs"]["commonroad-v1"]
    PATH_PARAMS are read from constants.py
    :return: a dict with the bound stuctured as: 'name_bound': [lower_b, upper_b]
    """
    config_file=PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Assume default environment configurations
    return config["sensititvity_analysis_bounds"]


def load_model(model_path: str, algo: str) -> BaseRLModel:
    """
    Load trained model

    :param model_path: Path to the trained model
    :param algo: The used RL algorithm
    """
    # Load the trained agent
    # TODO: load last model if best_model.zip does not exist (no evaluation env was created during training)
    files = os.listdir(model_path)
    if "best_model.zip" in files:
        model_path = os.path.join(model_path, "best_model.zip")
    else:
        # find last model
        files = sorted(glob.glob(os.path.join(model_path, "rl_model*.zip")))
        def extract_number(f):
            s = re.findall("\d+", f)
            return (int(s[-1]) if s else -1, f)
        model_path = max(files, key=extract_number)
    model = [ALGOS[algo].load(model_path)]

    return model


def load_all_models(model_path: str, algo: str) -> BaseRLModel:
    """
    Load trained model

    :param model_path: Path to the trained model
    :param algo: The used RL algorithm
    """
    # Load all Models
    def extract_number(f):
            s = re.findall("\d+", f)
            return (int(s[-1]) if s else -1, f)
        
    files = sorted(glob.glob(os.path.join(model_path, "rl_model*.zip")), key=extract_number)
    models = []
    for each in files:
        print(f"loading model {each}")
        models.append(ALGOS[algo].load(each))
    return models


def load_env(args):
    """
    This function loads an enviromenet based on the env_configs

    :param args: the arguments input through the parser, needed for model_path, data_path
    :return: a Commonroad Env
    """

    #Load Kwargs
    env_configs = {}
    with open(os.path.join(args.model_path, "environment_configurations.yml"), "r") as config_file:
        env_configs = yaml.safe_load(config_file)
    env_kwargs = env_configs  

    # Create environment
    # note that CommonRoadVecEnv is inherited from DummyVecEnv
    env_kwargs['test_reset_config_path'] = args.data_path + '//problem_test'
    env_kwargs['train_reset_config_path'] = args.data_path + '//problem_train'
    env_kwargs['meta_scenario_path'] = args.data_path + '//meta_scenario'
    env = CommonroadEnv(**env_kwargs)
    return env

def helper_compress_figure_labelx(labels, values):
    """
    This function makes the plot smaller (compresses the x axis of the plot) by taking the mean abs. value of multiple same type observations

    :param labels: list of labels, np.ndarray or 1D List of strings
    :param values: list of values, np.ndarray
    :return: Tuple(labels_new, values_new)
    """
    assert len(labels) == len(values), f"values: {values} and labels: {labels} need to be same lenght"
    # find unique labels and indices
    indexdict = {}
    for index, name_l in enumerate(labels):
        found_similar = False
        for exis in indexdict.keys():
            if name_l.startswith(exis):
                indexdict[exis].append(index) 
                found_similar=True
                break
        if not found_similar:
            indexdict[name_l]= [index]
    
    # new values
    values_new = []

    for indices in indexdict.values():
        values_new.append(np.average(np.abs(values[indices])))
    
    return list(indexdict.keys()), np.array(values_new)

def refactor_name_list(name_list):
    """
    This function removes the numbers of the observation names if the observation is a singular observation

    :param name_list: list of the names associated with the observations like ['[obs1]_00', [obs2]_00', '[obs2]_01', ...]
    """
    for i, each in enumerate(name_list):
        if each[-1] == '0' and each[-2] == '0':
            name_list[i] = each[:-3]
    return name_list


def plot_results_1dim(labels, values, meaning, path, ylim=(-0.1, 0.4), fformat="png"):
    """
    This function plots 1 dimensional resutls of a sensitivity analysis as a bar chart.
    This plot is saved at the in args specified location
    :param labels: the names associated with the individual observations as a list
    :param values: the values associated with the individual observations as a list in the same order as the labels
    :param meaning: a string describing the meaning of the plot
    :param path: save path
    """

    x_pos = [i for i, _ in enumerate(labels)]
    breite = len(labels) * 0.17
    plt.subplots(figsize=(breite, 8))
    plt.gcf().subplots_adjust(bottom=0.60)

    plt.bar(x_pos, values, color='green')

    plt.xlabel("Observations")
    plt.ylabel("Sensitivity value for each Obs")
    plt.title("Sensitivity analysis " + meaning)
    plt.xticks(x_pos, labels, rotation=90)

    plt.ylim(1.1 * ylim[0], 1.1 * ylim[1])


    plt.savefig(path + '/sens_analysis_' + meaning + '.' + fformat)
    plt.close()


def plot_results_2dim(labels, values, meaning, path, fformat="png"):
    """
    This function plots 2 dimensional resutls of a sensitivity analysis as a bar chart.
    This plot is saved at the in args specified location
    :param labels: the names associated with the individual observations as a list
    :param values: the values associated with the individual observations as a list in the same order as the labels
    :param meaning: a string describing the meaning of the plot
    :param path: save path    
    """

    for each in labels:
        if each[-1]  == '0' and each[-2] == '0':
            each = each[0:-3]  

    fig, ax = plt.subplots(figsize=(30, 30))
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, round(values[i, j],2),
                        ha="center", va="center", color="w")

    ax.set_title("Sensitivity analysis " + meaning)
    fig.tight_layout()
    plt.savefig(path + '/sens_analysis_' + meaning + '.' + fformat)
    plt.close()



def perform_analysis(args):
    """
    Performs the whole analysis workflow
    If args.save_fig is true, the plots are saved
    The single steps are given in inline comments
    :param args: :param args: parsed input arguments
    """
    # Load environement
    env = load_env(args)
    # Load pretrained model
    if args.save_gif == False:
        models = load_model(args.model_path, args.algo)
    else:
        models = load_all_models(args.model_path, args.algo)


    
    # add observation w observation bounds
    env.reset()
    obs = env.observation_dict
    num_vars = 0
    name_list =[]
    for key in obs:       
        num_vars += len(obs[str(key)])
        for i in range(len(obs[str(key)])):
            name_list.append(str(key + '_' + "{0:02}".format(i)))



    # get sample bounds
    bound_list = []
    bound_configs = open_configs_sens_bounds()
    for each in name_list:
        assert each[:-3] in bound_configs, str(each[:-3]) \
            + ' missing bound, please add in respective section of configs (on the bottom)' 
        bound_list.append(bound_configs[each[:-3]])

    problem = {
    'num_vars': num_vars,
    'names': name_list,
    'bounds': bound_list
    }

    # generate Samples
    param_values = SAMPLER[args.method].sample(problem, args.n)  

    
    # correct sampling errrors with boolean bounds
    for sample in param_values:
        for i in range(len(bound_list)):
            if True in bound_list[i]:
                sample[i] = np.random.choice(bound_list[i])
    
    if args.save_gif:
        pic_dict1 = {}
        pic_dict2 = {}
        for each in RESULTS[args.method]: 
                    pic_dict1[each] = []
                    pic_dict2[each] = []   
    values1 = []
    values2 = []

    for model_num, model in tqdm(enumerate(models), desc="Evaluation all Models"):
        print(f"performing analysis on model {model_num}  / {len(models)}")
        # for n defined in args collect result of model.predict
        Model_predictions = []
        for sample in param_values:
            y = model.predict(sample, deterministic=True)
            Model_predictions.append(y[0])
        Model_predictions = np.array(Model_predictions, dtype=float)
        Model_predictions = np.transpose(Model_predictions)
        Y = np.zeros([param_values.shape[0]])
        
        # Perform analysis
        if args.method == 'rbd_fast' or args.method == 'morris' or args.method == 'delta' or args.method == 'ff':
            Si = ANALYZERS[args.method].analyze(problem, param_values, Model_predictions[0], print_to_console=False)
            Si2 = ANALYZERS[args.method].analyze(problem,param_values, Model_predictions[1], print_to_console=False)
        else:
            Si = ANALYZERS[args.method].analyze(problem, Model_predictions[0], print_to_console=False)
            Si2 = ANALYZERS[args.method].analyze(problem, Model_predictions[1], print_to_console=False)

        # Output analysis results
        # Plot result

        name_list = refactor_name_list(name_list)

        if args.save_data and model == models[-1]:
            path = os.path.join(args.save_path, 'sens_analysis_data')
            
            os.makedirs(path, exist_ok=True)
            saved_arrays = {
                f"{args.method}_labels": np.asarray(name_list),
            }
            for k, v in Si.items():
                saved_arrays[f"{args.method}_action1_{k}"] = v
            for k, v in Si.items():
                saved_arrays[f"{args.method}_action2_{k}"] = v
            
            np.savez(os.path.join(path, f"sens_analysis_{args.method}"), **saved_arrays)

        if args.save_fig and model == models[-1]:
            print("saving figs")
            path = os.path.join(args.save_path, 'sens_analysis_figs')
            os.makedirs(path, exist_ok=True)

            for each in RESULTS[args.method]:
                meaning1 = str(args.method + "_action1_" + each)
                meaning2 = str(args.method + "_action2_" + each)  
                if each == 'S2' or each == 'IE':
                    plot_results_2dim(name_list, Si2[each], meaning2, path)
                    plot_results_2dim(name_list, Si[each], meaning1, path)
                else:
                    plot_results_1dim(name_list, Si[each], meaning1, path, (min(Si[each]), max(Si[each])))
                    plot_results_1dim(name_list, Si2[each], meaning2, path, (min(Si2[each]), max(Si2[each])))
        
        if args.save_gif:
            values1.append(Si)
            values2.append(Si2)

    if args.save_gif:
        path = os.path.join(args.save_path, 'sens_analysis_figs')

        for each in tqdm(RESULTS[args.method], desc="Constructing gifs"):
            max_value = max(max(value[each]) for value in values1)
            min_value = min(min(value[each]) for value in values1)
            for val_iter in tqdm(range(0, len(values1) - 1), desc=f"-Images for first action of {each}"):
                results = values1[val_iter][each]
                next_results = values1[val_iter + 1][each]
                dist_results = np.array(next_results) - np.array(results)
                for i in range(0, args.n_frames + 1):
                    interpol_results = (results + (dist_results/args.n_frames) * i)
                    meaning = str(args.method + "_action1_" + each) + "_" + str(val_iter) + "_" + str(i)
                    pic_dict1[each].append(path + '/sens_analysis_' + meaning + '.png')
                    if i == args.n_frames:
                        for j in range(0, 5):
                            pic_dict1[each].append(path + '/sens_analysis_' + meaning + '.png')
                    if each == 'S2' or each == 'IE':
                        plot_results_2dim(name_list, interpol_results, meaning, path)
                    else:
                        plot_results_1dim(name_list, interpol_results, meaning, path, ylim=(min_value, max_value))

            max_value = max(max(value[each]) for value in values2)
            min_value = min(min(value[each]) for value in values2)
            for val_iter in tqdm(range(0, len(values2) - 1), desc=f"-Images for second action {each}"):
                results = values2[val_iter][each]
                next_results = values2[val_iter + 1][each]
                dist_results = np.array(next_results) - np.array(results)
                for i in range(0, args.n_frames + 1):
                    interpol_results = (results + (dist_results/args.n_frames) * i)
                    meaning = str(args.method + "_action2_" + each) + "_" + str(val_iter) + "_" + str(i)
                    pic_dict2[each].append(path + '/sens_analysis_' + meaning + '.png')
                    if i == args.n_frames:
                        for j in range(0, 5):
                            pic_dict2[each].append(path + '/sens_analysis_' + meaning + '.png')
                    if each == 'S2' or each == 'IE':
                        plot_results_2dim(name_list, interpol_results, meaning, path)
                    else:
                        plot_results_1dim(name_list, interpol_results, meaning, path, ylim=(min_value, max_value))
            
            new_path = os.path.join(path, 'gifs')
            os.makedirs(path, exist_ok=True)

            gif_path  = new_path + "/" + each + '_action1.gif'           
            with imageio.get_writer(gif_path, mode = 'I') as writer:
                for filename in tqdm(pic_dict1[each], desc=f"-Gif for first action {each}"):
                    image = imageio.imread(filename)
                    writer.append_data(image)

            for filename in set(pic_dict1[each]):
                os.remove(filename)

            gif_path  = new_path + "/" + each + '_action2.gif'
            with imageio.get_writer(gif_path, mode = 'I') as writer:
                for filename in tqdm(pic_dict2[each], desc=f"-Gif for second action {each}"):
                    image = imageio.imread(filename)
                    writer.append_data(image)

            for filename in set(pic_dict2[each]):
                os.remove(filename)



def main():
    """
    Main method, calling the parser and the sensititvity analysis
    """
    args = get_parser().parse_args()
    args_assertions(args)
    perform_analysis(args)

if __name__ == "__main__":
    main()
