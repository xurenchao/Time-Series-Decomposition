import sys
sys.path.append('utils')
import os
import argparse
import json
from tensorboardX import SummaryWriter
from dataset.spot import DailyDataset_nn, get_loader, TOTAL_STD, TOTAL_MEAN, SPOT
from tool import to_gpu
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from pi_models import NN


def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=35, type=int,)
    parser.add_argument('--batch_size', default=64, type=int,)
    parser.add_argument('--N', default=2000, type=int,)
    parser.add_argument('--W', default=14, type=int,)
    parser.add_argument('--input_dim', default=14*24, type=int,)
    parser.add_argument('--output_dim', default=24, type=int,)
    parser.add_argument('--hidden_size', default=[128], type=int,)
    parser.add_argument('--flag', default='bootstrap', type=str,)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if args.flag is None:
        writer = SummaryWriter(log_dir=None)
        writer_path = list(writer.all_writers.keys())[0]
    else:
        writer_path = os.path.join('runs/pi', args.flag)
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    model = to_gpu(NN(input_dim=336, hidden=[128], output_dim=24, bn=0, dropout=0))
    nn.init.kaiming_normal_(model.layers.Linear0.weight)
    nn.init.kaiming_normal_(model.h2o.weight)
    dataset = DailyDataset_nn(N=2000, W=14)

    testX, testY = dataset.get_io(start_date='2016-01-01', end_date='2016-06-30')
    loader = get_loader(dataset, batch_size=64, shuffle=True, num_workers=1)
    with torch.no_grad():
        testX = to_gpu(testX)
        testY = to_gpu(testY) * TOTAL_STD

    criterion = nn.MSELoss()
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    seed_x = []
    seed_y = []
    for _, (x, y, item) in enumerate(loader):
        seed_x += [x]
        seed_y += [y]
    seed_x = seed_x[:12]
    seed_y = seed_y[:12]

    epochs = args.epochs
    global_step = 0
    for epoch in tqdm(range(epochs)):       
        train_out = to_gpu(torch.Tensor([]))
        train_y = to_gpu(torch.Tensor([]))
        
        for i in range(len(seed_x)):
            x = seed_x[i]
            y = seed_y[i]
            B, W, D = x.shape
            x = to_gpu(x.reshape(B, W*D))
            y = to_gpu(y)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            train_out = torch.cat((train_out, out), 0)
            train_y = torch.cat((train_y, y), 0)
            global_step += 1

        torch.save(model.state_dict(), os.path.join(writer_path, "snapshots%s.pth" % (epoch+1)))
            
        train_out = train_out * TOTAL_STD
        train_y = train_y * TOTAL_STD
        train_mse = criterion(train_out, train_y)**.5
        train_mae = metric(train_out, train_y)
        writer.add_scalar('train_mse', train_mse.data, global_step=(epoch+1))
        writer.add_scalar('train_mse', train_mse.data, global_step=(epoch+1))

        model.eval()
        y = []
        for i in range(182):
            x = testX[i:i+14].reshape(14*24)
            y += [model(x)]
        test_out = torch.stack(y, 0) * TOTAL_STD
        test_mse = criterion(test_out, testY)**.5
        test_mae = metric(test_out, testY)
        writer.add_scalar('test_mse', test_mse.data, global_step=(epoch+1))
        writer.add_scalar('test_mae', test_mae.data, global_step=(epoch+1))
        
        model.train()
    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
    np.savetxt('runs/pi/bootstrap/results/'+args.flag.split('/')[-1]+'.csv', test_out.data.cpu().numpy())
    


