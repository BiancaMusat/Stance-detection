import torch
import time
import datetime
import random
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from keras.utils import pad_sequences
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

class Model:
    """
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it.
    """
    def __init__(self, use_cuda=False):
        super(Model, self).__init__()

        # GPU support
        self.device = torch.device("cpu")
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda")
        

    def tokenize(self, model_name, premise_data, hypothesis_data, tokenizer, MAX_LEN):
        """
        Tokenize data
        """
        # add special tokens for to split the premise from the hypothesis
        sentences = ["[CLS] " + premise_data[i] + " [SEP]" + hypothesis_data[i] + "[SEP]" for i in range(0,len(premise_data))]
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # Pad input tokens
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        
        attention_masks = []  # Create attention masks
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # create a mask to differentiate the premise from the hypothesis
        token_type_ids = []
        for seq in input_ids:
            type_id = []
            condition = 'sent1'
            for i in seq:
                if condition == 'sent1':
                    type_id.append(0)
                    if i == 102:  # the separator [SEP] is encoded as 102
                        condition = 'sent2'
                elif condition == 'sent2':
                    type_id.append(1)
            token_type_ids.append(type_id)
            
        # Convert to torch tensors
        data_inputs = torch.tensor(input_ids, device = self.device)
        data_masks = torch.tensor(attention_masks, device = self.device)
        data_token_ids = torch.tensor(token_type_ids, device = self.device)
        return data_inputs, data_masks, data_token_ids


    def get_transformer_model(self, model_name):
        """
        Get tokenizer and model
        """
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels = 3,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        model.cuda()  # run model on GPU

        return tokenizer, model


    def get_optimizer_scheduler(self, model, train_dataloader_len, epochs):
        """
        Get optimizer and scheduler
        """
        optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
        total_steps = train_dataloader_len * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)
        return optimizer, scheduler


    def flat_accuracy(self, preds, labels):
        """
        Compute flat accuracy
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    ################ Time elapsed ####################
    ###############################################################

    def format_time(self, elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        return str(datetime.timedelta(seconds=int(round((elapsed)))))


    def train(self, epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler):
        """
        Train model
        """
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch
        loss_values = []

        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            # Reset the total loss for this epoch
            total_loss = 0

            # Put the model into training mode
            model.train()

            # For each batch of training data
            for step, batch in enumerate(train_dataloader):

                # Progress update every 64 batches.
                if step % 64 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0) # Calculate elapsed time in minutes
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from the dataloader
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_input_tokens = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)
                b_indexes = batch[4]

                # Clear any previously calculated gradients before performing a backward pass
                model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we have provided the labels
                outputs = model(b_input_ids,
                            token_type_ids=b_input_tokens,
                            attention_mask=b_input_mask,
                            labels=b_labels)
                
                # Get the loss value out of the tuple.
                loss = outputs[0]
            
                # Generate logits file
                with open("logits.txt", "a") as f:
                    f.write(str(epoch_i) + '\n')
                    f.write(str(step) + '\n')
                    f.write(str(list(b_indexes)).replace("\n", "") + '\n')
                    f.write(str(list(b_labels)).replace("\n", "") + '\n')
                    f.write(str(list(outputs[1])).replace("\n", "") + '\n')
            
                # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0, to help prevent the exploding gradients problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update the learning rate
                scheduler.step()

            # Calculate the average loss over the training data
            avg_train_loss = total_loss / len(train_dataloader)            
            
            # Store the loss value for plotting the learning curve
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))
                
            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode
            model.eval()

            # Tracking variables 
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)  # Add batch to GPU
                b_input_ids, b_input_mask, b_input_tokens, b_labels, b_indexes = batch  # Unpack the inputs from dataloader

                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions
                    # This will return the logits rather than the loss because we have not provided labels
                    outputs = model(b_input_ids, 
                                    token_type_ids=b_input_tokens, 
                                    attention_mask=b_input_mask)
                
                # Get the "logits" output by the model
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()
                
                # Calculate the accuracy for this batch of test sentences
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                
                # Accumulate the total accuracy
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

        print("")
        print("Training complete!")
        return model


    def evaluate(self, prediction_dataloader, model):
        """
        Evaluate model
        """

        print("Evaluating on the testset")

        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions, true_labels = [], []

        # Predict 
        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)  # Add batch to GPU
            b_input_ids, b_input_mask, b_input_tokens, b_labels, b_indexes = batch  # Unpack the inputs from the dataloader
            
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=b_input_tokens, 
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            
            # Store predictions and true labels
            predictions.extend(pred_flat)
            true_labels.extend(labels_flat)

        print('DONE.')

        print('Classification accuracy is')
        print(metrics.accuracy_score(true_labels, predictions)*100)
        print(classification_report(true_labels, predictions, target_names = ['neutral', 'for', 'against']))
        

    def predict(self, prediction_dataloader, model):
        """
        Predict labels
        """

        print('Predict labels')

        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions , true_labels, indexes = [], [], []

        # Predict 
        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)  # Add batch to GPU
            b_input_ids, b_input_mask, b_input_tokens, b_labels, b_indexes = batch  # Unpack the inputs from the dataloader
            
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=b_input_tokens, 
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            
            # Store predictions and true labels
            predictions.extend(pred_flat)
            true_labels.extend(labels_flat)

            # get instances' ids
            batch_indexes = str(list(b_indexes)).replace("\n", "")
            batch_ids = re.findall('\(.*?\,', batch_indexes[1:-2])
            batch_ids = [int(x[1:-1]) for x in batch_ids]
            indexes.extend(batch_ids)

        return predictions, indexes
    