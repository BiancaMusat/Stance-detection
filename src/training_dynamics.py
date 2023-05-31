import re
import argparse
import os
import shutil

def generate_dynamics(logits_file, EPOCHS):
    """
    Function that generates the training dynamics files
    logits_file has the following format:
        - for each epoch and each batch repeat:
            - epoch number
            - batch number
            - data instances indexes in batch
            - gold labels for the examples corresponding to the current epoch/batch
            - the loggits corresponding to the current epoch/batch
    EPOCHS is the number of epochs
    """
    all_data = []  # will contain all data from logits file
    with open(logits_file, "r") as f:
        for line in f:
            all_data.append(line)
    file_data = {x : [] for x in range(0, EPOCHS)}  # the logits corresponding to each epoch
    for i in range(0, len(all_data), 5):
        epoch = int(all_data[i])  # read the epoch
        batch = int(all_data[i + 1])  # read the batch number
        batch_ids = re.findall('\(.*?\,',all_data[i + 2][1:-2])  # read indexes
        batch_ids = [int(x[1:-1]) for x in batch_ids]
        labels = re.findall('\(.*?\,',all_data[i + 3][1:-2])  # read the gold labels
        labels = [int(x[1:-1]) for x in labels]
        all_logits = re.findall('\[.*?\]',all_data[i + 4][1:-2])  # read the logits

        logits = []
        for l in all_logits:
            logits.append([float(x) for x in l[1:-1].split(',')])  # generate the list of logits
        for j, id in enumerate(batch_ids):
                file_data[epoch].append({"guid" : id, "logits_epoch_"+str(epoch) : logits[j], "gold": labels[j]})  # enerate trainin dynamics

    if not os.path.exists('training_dynamics'):
        os.makedirs('training_dynamics')

    file_names = ["training_dynamics/dynamics_epoch_" + str(x) + ".jsonl" for x in range(0, EPOCHS)]
    for i, file_name in enumerate(file_names):
        with open(file_name, "w") as f:  # write the dynamics to the corresponding files
            for data in file_data[i]:
                f.write(str(data).replace('\'', '\"') + '\n')  # change single quotes to double quotes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--logits_file', help='Name of the logits file', required=False)
    parser.add_argument('-e', '--epochs', help='The number of epochs', required=False)
    args = vars(parser.parse_args())

    logits_file = "logits.txt"
    epochs = 4

    if args["logits_file"] is not None:
        logits_file = args["logits_file"]

    if args["epochs"] is not None:
        epochs = int(args["epochs"])

    # delete previously created dynamics
    if os.path.exists('training_dynamics'):
        shutil.rmtree('training_dynamics')

    generate_dynamics(logits_file, epochs)
    