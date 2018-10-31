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

from dataset.spot import DailyDataset, PeriodDataset, get_loader, TOTAL_STD
from utils.tool import clip_grads, to_gpu
from utils.gated_model import ResRNN as Model


def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=50, type=int,)
    parser.add_argument('--batch_size', default=64, type=int,)
    parser.add_argument('--N', default=2000, type=int,)
    parser.add_argument('--W', default=14, type=int,)
    parser.add_argument('--P', default=1, type=int,)
    parser.add_argument('--input_dim', default=24, type=int,)
    parser.add_argument('--output_dim', default=24, type=int,)
    parser.add_argument('--hidden_size', default=64, type=int,)
    parser.add_argument('--hard_gate', default=1, type=bool,)
    # parser.add_argument('--week_penalty', default=3e-5, type=float,)
    # parser.add_argument('--day_penalty', default=3e-4, type=float,)
    # parser.add_argument('--smooth_penalty', default=0.001, type=float,)
    parser.add_argument('--flag', default='res_rnn_2', type=str,)
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
        # writer_path = os.path.join('runs/seasonality', args.flag)
        writer_path = os.path.join('runs/test', args.flag)
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    dataset = PeriodDataset(N=args.N, W=args.W, P=args.P)
    loader = get_loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    TEST_LENGTH = 182

    trainX, trainY = loader.dataset.get_io('2011-06-30', '2015-12-31')
    testX, testY = loader.dataset.get_io('2011-06-30', '2016-06-30')

    # add trend
    add_trend=0
    if add_trend:
        a=[x for x in range(0, 24*182)]
        b=torch.zeros_like(testX)
        b[-182:,:]=torch.Tensor(a).reshape(182,24)*0.0002
        c=[x for x in range(1, 24*182+1)]
        d=torch.Tensor(c).reshape(182,24)*0.0002
    else:
        b=d=0

    with torch.no_grad():
        train_period_input = to_gpu(trainX)
        train_period_output = to_gpu(trainY) * TOTAL_STD
        test_period_input = to_gpu(testX+b)
        test_period_output = to_gpu(testY[-TEST_LENGTH:]+d) * TOTAL_STD

    model = to_gpu(Model(input_dim=args.input_dim, output_dim=args.output_dim,
                         hidden_size=args.hidden_size, hard_gate=args.hard_gate))

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

        writer.add_scalar('train_mse', train_mse.data, global_step=epoch)
        writer.add_scalar('train_mae', train_mae.data, global_step=epoch)
        writer.add_scalar('test_mse', test_mse.data, global_step=epoch)
        writer.add_scalar('test_mae', test_mae.data, global_step=epoch)

        for _, (x, y, item) in enumerate(loader):
            x = to_gpu(x)
            y = to_gpu(y)

            optimizer.zero_grad()

            out = model(x)
            # out, week_loss, day_loss, smooth_loss = model(x)

            loss = criterion(out, y) #+ args.week_penalty * week_loss + args.day_penalty * day_loss + args.smooth_penalty * smooth_loss
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            # writer.add_scalar('week_loss', week_loss.data, global_step=global_step)
            # writer.add_scalar('day_loss', day_loss.data, global_step=global_step)
            # writer.add_scalar('smooth_loss', smooth_loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            # clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % epoch))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
