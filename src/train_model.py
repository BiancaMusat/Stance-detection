import re
import argparse
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import model_utils


use_cuda = torch.cuda.is_available()

def get_data(df):
    """
    Load datasets
    """
    x1, x2, y = df['premise'], df['text'], df['stance']
    x1, x2, y = x1.tolist(), x2.tolist(), y.tolist()

    return (x1, x2, y)


def get_data_loader(batch_size, inputs, masks, token_ids, labels, indexes):
    """
    Create dataloader
    """
    data = TensorDataset(inputs, masks, token_ids, labels, indexes)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return data, sampler, dataloader


def main():
    # GPU support
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda")

    if os.path.exists("logits.txt"):  # remove previous logits file
        os.remove("logits.txt")

    # read aruments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Name of the model', required=False)
    parser.add_argument('-l', '--max_len', help='Length of the embeddings', required=False)
    parser.add_argument('-e', '--epochs', help='The number of epochs', required=False)
    parser.add_argument('-b', '--batch_size', help='The number of batches', required=False)
    parser.add_argument('-train', '--train_data', help='Path to train data', required=True)
    parser.add_argument('-test', '--test_data', help='Path to test data', required=True)
    parser.add_argument('-val', '--val_data', help='Path to validation data', required=True)
    args = vars(parser.parse_args())

    model_name = "digitalepidemiologylab/covid-twitter-bert-v2"
    max_len = 128
    epochs = 4
    batch_size = 32

    if args["model"] is not None:
        model_name = args["model"]
    
    if args["max_len"] is not None:
        max_len = args["max_len"]

    if args["epochs"] is not None:
        epochs = int(args["epochs"])

    if args["batch_size"] is not None:
        batch_size = int(args["batch_size"])

    # Loading the datasets
    train_df = pd.read_csv(args["train_data"])
    test_df = pd.read_csv(args["test_data"])
    val_df = pd.read_csv(args["val_data"]) 
    # weak_df = pd.read_csv('datasets/cstance_weak.csv')

    (x1_train, x2_train, y_train) = get_data(train_df)
    (x1_val, x2_val, y_val) = get_data(val_df)
    (x1_test, x2_test, y_test) = get_data(test_df)
    # (x1_weak, x2_weak, y_weak) = get_data(weak_df)

    # get tokenizer and model
    model_handler = model_utils.Model(use_cuda)
    tokenizer, model = model_handler.get_transformer_model(model_name)

    # tokenize data
    train_inputs, train_masks, train_token_ids = model_handler.tokenize(model_name, x1_train, x2_train, tokenizer, max_len)
    val_inputs, val_masks, val_token_ids = model_handler.tokenize(model_name, x1_val, x2_val, tokenizer, max_len)
    test_inputs, test_masks, test_token_ids = model_handler.tokenize(model_name, x1_test, x2_test, tokenizer, max_len)
    # weak_inputs, weak_masks, weak_token_ids = model_handler.tokenize(model_name, x1_weak, x2_weak, tokenizer, max_len)

    # convert the labels into torch tensors
    train_labels = torch.tensor(y_train, dtype=torch.long, device = device)
    val_labels = torch.tensor(y_val, dtype=torch.long, device = device)
    test_labels = torch.tensor(y_test, dtype=torch.long, device = device)
    # weak_labels = torch.tensor(y_weak, dtype=torch.long, device = device)

    # generate data indexes (to keep track of the instance's id)
    train_indexes = torch.tensor(range(len(y_train)), dtype=torch.long, device = device)
    val_indexes = torch.tensor(range(len(y_val)), dtype=torch.long, device = device)
    test_indexes = torch.tensor(range(len(y_test)), dtype=torch.long, device = device)
    # weak_indexes = torch.tensor(range(len(y_weak)), dtype=torch.long, device = device)

    # Get the dataloaders
    train_data, train_sampler, train_dataloader = get_data_loader(batch_size, train_inputs, train_masks, train_token_ids, train_labels, train_indexes)
    val_data, val_sampler, val_dataloader = get_data_loader(batch_size, val_inputs, val_masks, val_token_ids, val_labels, val_indexes)
    test_data, test_sampler, test_dataloader = get_data_loader(batch_size, test_inputs, test_masks, test_token_ids, test_labels, test_indexes)
    # weak_data, weak_sampler, weak_dataloader = get_data_loader(batch_size, weak_inputs, weak_masks, weak_token_ids, weak_labels, weak_indexes)

    # Get optimizer and scheduler
    optimizer, scheduler = model_handler.get_optimizer_scheduler(model, len(train_dataloader), epochs)

    # Train model
    model = model_handler.train(epochs, model, train_dataloader, val_dataloader, optimizer, scheduler)

    # Eval model
    model_handler.evaluate(test_dataloader, model)

    # # Predict on Weak set
    # predictions, indexes = model_handler.predict(weak_dataloader, model)
    # with open("predictions.txt", "a") as f:
    #     f.write(str(predictions) + '\n')
    #     f.write(str(indexes) + '\n')

if __name__ == "__main__":
    main()