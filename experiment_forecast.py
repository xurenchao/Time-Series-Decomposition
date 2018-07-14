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

from dataset.spot import PeriodDataset, get_loader, TOTAL_STD
from utils.tool import to_gpu
# from utils.gated_model import GatedRNN as Model
# from utils.gated_model import StackGatedRNN as Model
from utils.gated_model import ResRNN as Model
# from utils.gated_model import StackResRNN as Model


def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=50, type=int,)
    parser.add_argument('--batch_size', default=64, type=int,)
    parser.add_argument('--N', default=2000, type=int,)
    parser.add_argument('--W', default=14, type=int,)
    parser.add_argument('--P', default=7, type=int,)
    parser.add_argument('--input_dim', default=168, type=int,)
    parser.add_argument('--output_dim', default=24, type=int,)
    parser.add_argument('--hidden_size', default=128, type=int,)
    parser.add_argument('--hard_gate', default=1, type=bool,)
    parser.add_argument('--flag', default='resrnn_test', type=str,)
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
        writer_path = os.path.join('runs/p7', args.flag)
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    dataset = PeriodDataset(N=args.N, W=args.W, P=args.P)
    loader = get_loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    TEST_LENGTH = 182
    FORE_LENGTH = 61

    trainX, trainY = loader.dataset.get_io('2012-01-01', '2015-12-31')
    testX, testY = loader.dataset.get_io('2012-01-01', '2016-06-30')
    self_testX, self_testY = loader.dataset.get_io('2012-01-01', '2016-08-30')
    with torch.no_grad():
        train_period_input = to_gpu(trainX)
        train_period_output = to_gpu(trainY) * TOTAL_STD
        test_period_input = to_gpu(testX)
        test_period_output = to_gpu(testY[-TEST_LENGTH:]) * TOTAL_STD
        # add one more day
        # self_test_input = test_period_input[:train_period_input.size()[0] + 1]
        self_test_input = test_period_input
        self_test_output = to_gpu(self_testY[-FORE_LENGTH:]) * TOTAL_STD
        

    model = to_gpu(Model(input_dim=args.input_dim, output_dim=args.output_dim, hidden_size=args.hidden_size, hard_gate=args.hard_gate))


    criterion = nn.MSELoss()
    metric = nn.L1Loss()
    regularization = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    epochs = args.epochs
    global_step = 0
    for epoch in tqdm(range(epochs)):
        train_period_forecast = model.forecast(train_period_input)[0] * TOTAL_STD
        train_mse = criterion(train_period_forecast, train_period_output)**.5
        train_mae = metric(train_period_forecast, train_period_output)

        test_period_forecast = model.forecast(test_period_input)[0][-TEST_LENGTH:] * TOTAL_STD
        test_mse = criterion(test_period_forecast, test_period_output)**.5
        test_mae = metric(test_period_forecast, test_period_output)

        self_test_forecast = model.self_forecast(self_test_input, step=TEST_LENGTH)[-TEST_LENGTH:] * TOTAL_STD
        self_test_mse = criterion(self_test_forecast, test_period_output)**.5
        self_test_mae = metric(self_test_forecast, test_period_output)

        writer.add_scalar('train_mse', train_mse.data, global_step=epoch)
        writer.add_scalar('train_mae', train_mae.data, global_step=epoch)
        writer.add_scalar('test_mse', test_mse.data, global_step=epoch)
        writer.add_scalar('test_mae', test_mae.data, global_step=epoch)
        # writer.add_scalar('self_test_mse', self_test_mse.data, global_step=epoch)
        # writer.add_scalar('self_test_mae', self_test_mae.data, global_step=epoch)

        for _, (x, y, item) in enumerate(loader):
            x = to_gpu(x)
            y = to_gpu(y)

            optimizer.zero_grad()

            out = model(x)
            # with torch.no_grad():
            #     y_mean = to_gpu(torch.ones_like(out[1]) * (out[1].mean()))

            loss = criterion(out, y)  # + 0.1*regularization(out[1], y_mean)
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            # clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % epoch))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
