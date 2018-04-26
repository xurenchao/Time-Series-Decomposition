import sys
sys.path.append('../utils')
# sys.path.append('../dataset0')
sys.path.append('../dataset')
import os
import argparse
from tqdm import tqdm
import json
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# from spot import DailyDataset, TOTAL_STD
from data_loader import PeriodDataset
from tool import to_var
# from gated_model import GatedRNN as Model
from basic_model import BasicRNN as Model

TOTAL_STD = 1


def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=50, type=int,)
    parser.add_argument('--batch_size', default=3, type=int,)
    parser.add_argument('--N', default=39, type=int,)
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
    TEST_LENGTH = 60
    # trainX, trainY = loader.dataset.get_io('2012-01-01', '2015-12-31')
    # testX, testY = loader.dataset.get_io('2012-01-01', '2016-06-30')
    trainX, trainY = loader.dataset.get_io('1977-01-01', '2012-12-31')
    testX, testY = loader.dataset.get_io('2012-01-01', '2017-12-31')
    train_period_input = to_var(trainX, volatile=True)
    train_period_output = to_var(trainY, volatile=True) * TOTAL_STD
    test_period_input = to_var(testX, volatile=True)
    # add one more day
    # self_test_input = test_period_input[:train_period_input.size()[0] + 1]
    test_period_output = to_var(testY[-TEST_LENGTH:], volatile=True) * TOTAL_STD
    # print(train_period_input.shape, train_period_output.shape)
    # print(test_period_input.shape, test_period_output.shape, self_test_input.shape)
    model = Model(seq_dim=args.seq_dim, hidden_size=args.hidden_size)
    model = model.cuda()
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
        # print(test_period_forecast.shape)
        test_mse = criterion(test_period_forecast, test_period_output)**.5
        test_mae = metric(test_period_forecast, test_period_output)

        # self_test_forecast = model.self_forecast(self_test_input, to_var(testY, volatile=True) * TOTAL_STD, step=TEST_LENGTH) * TOTAL_STD
        # print(self_test_forecast.shape)
        # self_test_mse = criterion(self_test_forecast, test_period_output)**.5
        # self_test_mae = metric(self_test_forecast, test_period_output)

        writer.add_scalar('train_mse', train_mse.data[0], global_step=epoch)
        writer.add_scalar('train_mae', train_mae.data[0], global_step=epoch)
        writer.add_scalar('test_mse', test_mse.data[0], global_step=epoch)
        writer.add_scalar('test_mae', test_mae.data[0], global_step=epoch)
        # writer.add_scalar('self_test_mse', self_test_mse.data[0], global_step=epoch)
        # writer.add_scalar('self_test_mae', self_test_mae.data[0], global_step=epoch)

        for _, (x, y, item) in enumerate(loader):
            x = to_var(x, requires_grad=False)
            y = to_var(y, requires_grad=False)
            # idx = to_var(item, requires_grad=False)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            writer.add_scalar('train_loss',
                              loss.data[0], global_step=global_step)
            loss.backward(retain_graph=True)
            # clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % epoch))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
