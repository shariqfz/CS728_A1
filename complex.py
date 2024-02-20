from eval import test
from plot import plot_history

import argparse
import json

from torch_geometric.nn import ComplEx
import torch_geometric
print("torch_geometric version: ", torch_geometric.__version__)


import torch
import torch.optim as optim
torch.__version__
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}\n")


###############################################################################

# Define global vars
temp_lr=0.001
temp_epochs = 5000
temp_freq = 25
temp_dataset_name = 'WN18RR_torch'
torch.manual_seed(1)
watch = True


EPOCHS = temp_epochs
history_freq = temp_freq
dataset_name = temp_dataset_name       # 'WN18RR_torch' | 'FB15k_torch'
lr=temp_lr

# Load data from local storage
path = f"./data/{dataset_name}/"

train_data = torch.load(path + 'train.pt')
valid_data = torch.load(path + 'valid.pt')
test_data  = torch.load(path + 'test.pt')

# Send data to GPU if available
train_data = train_data.to(device)
valid_data = valid_data.to(device)
test_data = test_data.to(device)



model = ComplEx(
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=512,    # earlier it was 50
    ).to(device)


loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)


optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-6) # default lr = 0.001


def train():
    model.train()
    total_loss = total_examples = 0
    cnt = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
        cnt += 1
    return total_loss / total_examples

history = {'rank':[], 'mrr':[], 'hits':[], 'map_sk':[], 'map_scr':[]}

print(f"model.num_nodes: {model.num_nodes}\n")
print('Epochs: ', end='')
for epoch in range(1, EPOCHS + 1):
    loss = train()
    
    print(f'{epoch:03d} ', end='')
    if epoch % history_freq == 0:
        
        # Evaluate on validation data
        mean_rank, mrr, hits, map_sk, map_scr = test(model, valid_data)
        
        history['rank'].append(mean_rank)
        history['mrr'].append(mrr)
        history['hits'].append(hits)
        history['map_sk'].append(map_sk)
        history['map_scr'].append(map_scr)
        
        print(f'\nEpoch: {epoch:03d}, Val Mean Rank: {mean_rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}, MAP_sk: {map_sk:.4f}, MAP_scr: {map_scr:.4f}')
        print()
        print('Epochs: ', end='') if epoch != EPOCHS else None

        # save history to file
        if watch:
            with open(f"./history/history_epochs_{EPOCHS}_{history_freq}_{dataset_name}.json", "w") as outfile: 
                json.dump(history, outfile)



# Evaluate on TEST data
mean_rank, mrr, hits_at_10, map_sk, map_scr = test(model, test_data)
print(f'\nTest Mean Rank: {mean_rank:.2f}, Test MRR: {mrr:.4f}, Test Hits@10: {hits_at_10:.4f}')

# save the model
torch.save(model, f'./checkpoints/{dataset_name}_{EPOCHS}_epochs_{str(lr)[2:]}_lr_complEx_channels_512.pth')

# load a model:
# model = ComplEx()
# model.load_state_dict(torch.load('saved_model.pth'))

# Plot metrics vs epochs
if watch:
    plot_history(dataset_name, {k:v for k,v in history.items() if k not in ['rank', 'map_scr', 'map_sk']}, history_freq)
    # plot_history(dataset_name, {'rank': history['rank']}, history_freq)





