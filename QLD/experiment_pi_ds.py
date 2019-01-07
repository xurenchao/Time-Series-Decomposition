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

from dataset.spot import DailyDataset_pi, get_loader, TOTAL_STD, TOTAL_MEAN, SEASONAL
from utils.tool import clip_grads, to_gpu, picp, mpiw, rpiw, cwc, MyLoss
from utils.pi_models import GRU as Model


def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=200, type=int,)
    parser.add_argument('--batch_size', default=64, type=int,)
    parser.add_argument('--N', default=1600, type=int,)
    parser.add_argument('--W', default=21, type=int,)
    parser.add_argument('--input_dim', default=24, type=int,)
    parser.add_argument('--hidden_size', default=64, type=int,)
    parser.add_argument('--output_dim', default=24*2, type=int,)
    parser.add_argument('--lmd', default=0.001, type=float,)
    parser.add_argument('--alpha', default=0.05, type=float,)
    parser.add_argument('--s', default=50, type=int,)
    parser.add_argument('--seasonal', default=1, type=bool,)
    parser.add_argument('--flag', default='drnn-lube', type=str,)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()

    if args.flag is None:
        writer = SummaryWriter(log_dir=None)
        writer_path = list(writer.all_writers.keys())[0]
    else:
        # e.g. 'runs/gru/hidden64'
        writer_path = os.path.join('runs/pi', args.flag+'_'+str(args.hidden_size)+\
                                   '_'+str(args.s)+'_'+str(args.alpha)+'_'+str(args.lmd))
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    dataset = DailyDataset_pi(N=args.N, W=args.W, seasonal=args.seasonal)
    loader = get_loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    TEST_LENGTH = 182

    seasonal = SEASONAL['2011-12-12':'2016-06-30'].values
    trainY_S = seasonal[:-182].reshape(-1)
    testY_S = seasonal[-182:].reshape(-1)
    trainX, trainY = loader.dataset.get_io('2012-01-01', '2015-12-31')
    testX, testY = loader.dataset.get_io('2012-01-01', '2016-06-30')

    print(testX.shape)
    print(testY.shape)
    train_period_output = (trainY.data.numpy().reshape(-1) + trainY_S) * TOTAL_STD
    test_period_output = (testY[-TEST_LENGTH:].data.numpy().reshape(-1) + testY_S) * TOTAL_STD
    with torch.no_grad():
        train_period_input = to_gpu(trainX)      
        test_period_input = to_gpu(testX)
        


    model = to_gpu(Model(input_dim=args.input_dim, hidden_size=args.hidden_size, output_dim=args.output_dim))

    Loss = MyLoss(lmd=args.lmd, alpha=args.alpha, s=args.s)
    criterion = nn.MSELoss()
    metric = nn.L1Loss()
    regularization = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    epochs = args.epochs
    global_step = 0
    for epoch in tqdm(range(epochs)):

        train_period_forecast = model.forecast(train_period_input)[0] * TOTAL_STD
        train_lower_bound = train_period_forecast[:, :24].data.cpu().numpy().reshape(-1) + trainY_S * TOTAL_STD
        train_upper_bound = train_period_forecast[:, 24:].data.cpu().numpy().reshape(-1) + trainY_S * TOTAL_STD
        train_picp = picp(train_period_output, train_lower_bound, train_upper_bound)
        train_nmpiw = mpiw(train_period_output, train_lower_bound, train_upper_bound, norm=1)
        train_nrpiw = rpiw(train_period_output, train_lower_bound, train_upper_bound, norm=1)
        train_cwc_95 = cwc(train_period_output, train_lower_bound, train_upper_bound, alpha=0.05)
        train_cwc_99 = cwc(train_period_output, train_lower_bound, train_upper_bound, alpha=0.01)

        writer.add_scalar('train_picp', train_picp, global_step=epoch)
        writer.add_scalar('train_nmpiw', train_nmpiw, global_step=epoch)
        writer.add_scalar('train_nrpiw', train_nrpiw, global_step=epoch)
        writer.add_scalar('train_cwc_95', train_cwc_95, global_step=epoch)
        writer.add_scalar('train_cwc_99', train_cwc_99, global_step=epoch)

        test_period_forecast = model.forecast(test_period_input)[0][-TEST_LENGTH:] * TOTAL_STD
        test_lower_bound = test_period_forecast[:, :24].data.cpu().numpy().reshape(-1) + testY_S * TOTAL_STD
        test_upper_bound = test_period_forecast[:, 24:].data.cpu().numpy().reshape(-1) + testY_S * TOTAL_STD
        test_picp = picp(test_period_output, test_lower_bound, test_upper_bound)
        test_nmpiw = mpiw(test_period_output, test_lower_bound, test_upper_bound, norm=1)
        test_nrpiw = rpiw(test_period_output, test_lower_bound, test_upper_bound, norm=1)
        test_cwc_95 = cwc(test_period_output, test_lower_bound, test_upper_bound, alpha=0.05)
        test_cwc_99 = cwc(test_period_output, test_lower_bound, test_upper_bound, alpha=0.01)

        writer.add_scalar('test_picp', test_picp, global_step=epoch)
        writer.add_scalar('test_nmpiw', test_nmpiw, global_step=epoch)
        writer.add_scalar('test_nrpiw', test_nrpiw, global_step=epoch)
        writer.add_scalar('test_cwc_95', test_cwc_95, global_step=epoch)
        writer.add_scalar('test_cwc_99', test_cwc_99, global_step=epoch)


        for _, (x, y, item) in enumerate(loader):
            x = to_gpu(x)
            y = to_gpu(y)

            optimizer.zero_grad()

            out = model(x)

            loss = Loss(out, y)
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % (epoch+1)))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
