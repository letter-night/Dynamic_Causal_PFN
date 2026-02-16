import pickle
import random

import numpy as np
import torch
import torchcde


def get_processed_data(raw_sim_data, scaling_params):
    """
    It takes the raw simulation data and the scaling parameters, and returns a dictionary with the
    following keys:

    - `current_covariates`: the current covariates (cancer volume and patient type)
    - `time_covariates`: the time covariates (intensity)
    - `previous_treatments`: the previous treatments (one-hot encoded)
    - `current_treatments`: the current treatments (one-hot encoded)
    - `outputs`: the outputs (cancer volume)
    - `active_entries`: the active entries (1 if the patient is still alive, 0 otherwise)
    - `unscaled_outputs`: the unscaled outputs (cancer volume)
    - `input_means`: the input means (cancer volume, patient type, chemo application, radio application)
    - `inputs_stds`: the input standard deviations (cancer volume,

    raw_sim_data (dict): the dataframe containing the simulation data
    scaling_params (tuple): the mean and standard deviation of the cancer volume and patient types

    CODE ADAPTED FROM: https://github.com/ioanabica/Counterfactual-Recurrent-Network
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    mean["chemo_application"] = 0
    mean["radio_application"] = 0
    std["chemo_application"] = 1
    std["radio_application"] = 1

    input_means = mean[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()
    input_stds = std[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()

    # Continuous values
    cancer_volume = (raw_sim_data["cancer_volume"] - mean["cancer_volume"]) / std[
        "cancer_volume"
    ]
    patient_types = (raw_sim_data["patient_types"] - mean["patient_types"]) / std[
        "patient_types"
    ]

    patient_types = np.stack(
        [patient_types for t in range(cancer_volume.shape[1])],
        axis=1,
    )

    # Continuous values
    intensity = raw_sim_data["intensity"]

    # Binary application
    chemo_application = raw_sim_data["chemo_application"]
    radio_application = raw_sim_data["radio_application"]
    sequence_lengths = raw_sim_data["sequence_lengths"]

    # Convert treatments to one-hot encoding

    treatments = np.concatenate(
        [
            chemo_application[:, :-offset, np.newaxis],
            radio_application[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 1
            ):
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 1
            ):
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

    one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

    current_covariates = np.concatenate(
        [
            cancer_volume[:, :-offset, np.newaxis],
            patient_types[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )

    time_covariates = np.concatenate(
        [
            intensity[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )
    outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[["cancer_volume"]].values.flatten()[
        0
    ]  # because we only need scalars here
    output_stds = std[["cancer_volume"]].values.flatten()[0]

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

    raw_sim_data["current_covariates"] = current_covariates
    raw_sim_data["time_covariates"] = time_covariates
    raw_sim_data["previous_treatments"] = one_hot_previous_treatments
    raw_sim_data["current_treatments"] = one_hot_treatments
    raw_sim_data["outputs"] = outputs
    raw_sim_data["active_entries"] = active_entries

    raw_sim_data["unscaled_outputs"] = (
        outputs * std["cancer_volume"] + mean["cancer_volume"]
    )
    raw_sim_data["input_means"] = input_means
    raw_sim_data["inputs_stds"] = input_stds
    raw_sim_data["output_means"] = output_means
    raw_sim_data["output_stds"] = output_stds

    return raw_sim_data


def process_data(pickle_map):
    """
    Returns processed train, val, test data from pickle map

    Args:
    pickle_map (dict): dict containing data from pickle map

    Returns:
    training_processed (np array): training data processed numpy
    validation_processed (np array): validation data processed numpy
    test_processed (np array): test data processed numpy
    """
    # load data from pickle_map
    training_data = pickle_map["training_data"]
    validation_data = pickle_map["validation_data"]
    test_data = pickle_map["test_data"]
    scaling_data = pickle_map["scaling_data"]

    # get processed data
    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)
    test_processed = get_processed_data(test_data, scaling_data)

    return training_processed, validation_processed, test_processed


def get_treatment_indices(tr):
    """
    It takes in a treatment matrix and returns the indices of the treated and untreated patients

    tr: the treatment matrix
    return: The indices of the treated and untreated patients.
    """
    all_counts = []

    for i in range(tr.shape[0]):
        count = 0
        for j in range(tr.shape[1]):
            comparison = tr[i][j] == np.array([1, 0, 0, 0])
            if comparison.all() == True:
                count += 1

        all_counts.append(count)

    untreated_indices = []
    treated_indices = []

    for idx, sum_vals in enumerate(all_counts):
        if sum_vals == 59:
            untreated_indices.append(idx)
        else:
            treated_indices.append(idx)

    return treated_indices, untreated_indices


def data_to_torch_tensor(data, sample_prop=1, time_concat=-1):
    """
    Returns torch tensors of data -- one step ahead

    Args:
    data (numpy array): np array containing data
    sample_prop (int): proportion of samples

    Returns:
    data_X (torch tensor): containing covariates
    data_A (torch tensor): containing previous_treatments
    data_y (torch tensor): containing outcomes
    data_tr (torch tensor): containing current treatments
    """

    # extract data
    data_x = data["current_covariates"]

    data_x = np.concatenate(
        (data["current_covariates"], data["time_covariates"]),
        axis=2,
    )

    # np.concatenate((training_processed['current_covariates'][:,0:n,:], training_processed['previous_treatments']), axis=2)
    data_a = data["previous_treatments"]

    data_time = None  # Because we include time in data_x

    data_y = np.concatenate((data["outputs"], data["active_entries"]), axis=2)
    data_tr = data["current_treatments"]

    treated_indices, untreated_indices = get_treatment_indices(tr=data_tr)

    total_samples = data_x.shape[0]

    # get samples based on sampling proportion
    sample_prop = 1
    samples = int(total_samples * sample_prop)

    # numpy to torch tensor
    sample_idxs = random.sample(range(0, total_samples), samples)

    data_X = torch.from_numpy(data_x[:, 0:58, :time_concat])
    data_A = torch.from_numpy(data_a[:, :, :])
    data_Time = None  # Because we include time in data_x
    data_y = torch.from_numpy(data_y[:, :, :])
    data_tr = torch.from_numpy(data_tr[:, :, :])

    return (
        data_X,
        data_A,
        data_Time,
        data_y,
        data_tr,
        treated_indices,
        untreated_indices,
    )


def write_to_file(contents, filename):
    """
    It takes in a variable called contents and a variable called filename, and
    then writes the contents to a pickle file with the name filename.

    contents (str): the data to be written to the file
    filename (str): the name of the file to write to
    """
    # write contents to pickle file

    with open(filename, "wb") as handle:
        pickle.dump(contents, handle)


def read_from_file(filename):
    """
    It loads the file from pickle.

    filename (str): the name of the file to read from
    return: A list of dictionaries.
    """
    # load file from pickle

    return pickle.load(open(filename, "rb"))