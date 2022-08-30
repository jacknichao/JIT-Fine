from baselines.CC2Vec.jit_utils import mini_batches, save
import torch
import os, datetime
import torch 
import torch.nn as nn
from tqdm import tqdm
from baselines.CC2Vec.jit_cc2ftr_model import HierachicalRNN


def train_model(data, params):
    pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code = data
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=pad_msg_labels, 
                            mini_batch_size=params.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    use_gpu = False
    
    # params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.save_dir = os.path.join(params.save_dir, params.project_name)
    print(params.save_dir)
    params.vocab_code = len(dict_code)    
    
    if len(pad_msg_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = pad_msg_labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # uncomment to use GPU
    # params.device = 'cpu' # uncomment to use CPU
    print("building model")
    model = HierachicalRNN(args=params)
    model.share_memory()
#     if torch.cuda.is_available():
    if params.device != 'cpu':  
        model = model.cuda()
        use_gpu = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCEWithLogitsLoss()
    # batches = batches[:10]
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for i, (batch) in enumerate(tqdm(batches)):
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word(use_gpu)
            state_sent = model.init_hidden_sent(use_gpu)
            state_hunk = model.init_hidden_hunk(use_gpu)

            pad_added_code, pad_removed_code, labels = batch
            
            if use_gpu:
                labels = torch.cuda.FloatTensor(labels)
            else:
                labels = torch.FloatTensor(labels)
            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
            loss = criterion(predict, labels)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        # log_writer.add_scalar("cc2ftr Train/Loss",float(total_loss), epoch)
        save(model, params.save_dir, 'epoch', epoch)