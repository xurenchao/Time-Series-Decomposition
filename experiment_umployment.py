import sys
sys.path.append('utils')
import os
import argparse
from tqdm import tqdm
import json
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# from dataset0.spot import DailyDataset, TOTAL_STD
from dataset.data_loader import PeriodDataset
from utils.tool import to_gpu
from utils.gated_model import GatedRNN as Model
# from utils.basic_model import BasicRNN as Model



TOTAL_STD = 1



def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=60, type=int,)
    parser.add_argument('--batch_size', default=2, type=int,)
    parser.add_argument('--N', default=34, type=int,)
    parser.add_argument('--W', default=2, type=int,)
    parser.add_argument('--seq_dim', default=12, type=int,)
    parser.add_argument('--hidden_size', default=24, type=int,)
    parser.add_argument('--flag', default=None, type=str,)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    # import pdb
    # pdb.set_trace()

    if args.flag is None:
        writer = SummaryWriter(log_dir=None)
        writer_path = list(writer.all_writers.keys())[0]
    else:
        # e.g. 'runs/gru/hidden64'
        writer_path = os.path.join('runs', args.flag)
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    # loader = DailyDataset.get_loader(batch_size=args.batch_size, N=args.N, W=args.W)
    loader = PeriodDataset.get_loader(batch_size=args.batch_size, N=args.N, W=args.W)

    # TEST_LENGTH = 182
    TEST_LENGTH = 5
    # trainX, trainY = loader.dataset.get_io('2012-01-01', '2015-12-31')
    # testX, testY = loader.dataset.get_io('2012-01-01', '2016-06-30')
    trainX, trainY = loader.dataset.get_io('1977-01-01', '2012-12-31')
    testX, testY = loader.dataset.get_io('1977-01-01', '2017-12-31')
    with torch.no_grad():
        train_period_input = to_gpu(trainX)
        train_period_output = to_gpu(trainY) * TOTAL_STD
        test_period_input = to_gpu(testX)
        # add one more day
        self_test_input = test_period_input[:train_period_input.size()[0] + 1]
        test_period_output = to_gpu(testY[-TEST_LENGTH:]) * TOTAL_STD

    model = to_gpu(Model(seq_dim=args.seq_dim, hidden_size=args.hidden_size))

    criterion = nn.MSELoss()
    metric = nn.L1Loss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    epochs = args.epochs
    global_step = 0
    for epoch in tqdm(range(epochs)):
        train_period_forecast = model.forecast(train_period_input) * TOTAL_STD
        train_mse = criterion(train_period_forecast, train_period_output)**.5
        train_mae = metric(train_period_forecast, train_period_output)

        test_period_forecast = model.forecast(test_period_input)[-TEST_LENGTH:] * TOTAL_STD
        test_mse = criterion(test_period_forecast, test_period_output)**.5
        test_mae = metric(test_period_forecast, test_period_output)

        self_test_forecast = model.self_forecast(self_test_input, step=TEST_LENGTH) * TOTAL_STD
        self_test_mse = criterion(self_test_forecast, test_period_output)**.5
        self_test_mae = metric(self_test_forecast, test_period_output)

        writer.add_scalar('train_mse', train_mse.data, global_step=epoch)
        writer.add_scalar('train_mae', train_mae.data, global_step=epoch)
        writer.add_scalar('test_mse', test_mse.data, global_step=epoch)
        writer.add_scalar('test_mae', test_mae.data, global_step=epoch)
        writer.add_scalar('self_test_mse', self_test_mse.data, global_step=epoch)
        writer.add_scalar('self_test_mae', self_test_mae.data, global_step=epoch)

        for _, (x, y, item) in enumerate(loader):
            x = to_gpu(x)
            y = to_gpu(y)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            # clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % epoch))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
