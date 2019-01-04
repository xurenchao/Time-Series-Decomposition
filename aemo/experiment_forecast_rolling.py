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

from dataset.spot import DailyDataset, get_loader, TOTAL_STD, TOTAL_MEAN
from utils.tool import clip_grads, to_gpu
from decompose_model import TDec_RNN as DModel
from utils.forecast_model import RNN as Model

# decomposition

dataset1 = DailyDataset(N=2000, W=21)

testX1, testY1 = dataset1.get_io('2011-06-30', '2016-06-30')
_, self_testY = dataset1.get_io('2016-07-01', '2016-08-31')
with torch.no_grad():
    test_period_input1= to_gpu(testY1)
    self_test_output = to_gpu(self_testY[-62:]) * TOTAL_STD
PATH1 = './runs/decompose'
run1 = 'T_gru_h_24_soft_1e-2_3e-6_1e-2'
model1 = to_gpu(DModel(input_dim=24, output_dim=24, hidden_size=24, cell='gru', hard_gate=0))
snap1 = 'snapshots35'
model1.load_state_dict(torch.load(os.path.join(PATH1, run1, snap1+'.pth')))
test_period_output1 = model1.forecast(test_period_input1)
cache = test_period_output1[2][-7:]

def get_args():
    parser = argparse.ArgumentParser(description='Experimental setting.')
    parser.add_argument('--epochs', default=50, type=int,)
    parser.add_argument('--batch_size', default=64, type=int,)
    parser.add_argument('--N', default=1600, type=int,)
    parser.add_argument('--W', default=14, type=int,)
    parser.add_argument('--input_dim', default=24*2, type=int,)
    parser.add_argument('--hidden_size', default=64, type=int,)
    parser.add_argument('--output_dim', default=24, type=int,)
    parser.add_argument('--seasonal', default=1, type=bool,)
    parser.add_argument('--flag', default='rnn-s', type=str,)
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
        writer_path = os.path.join('runs/forecast', args.flag)
        writer = SummaryWriter(log_dir=writer_path)

    with open(os.path.join(writer_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    dataset = DailyDataset(N=args.N, W=args.W, seasonal=args.seasonal)
    loader = get_loader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    TEST_LENGTH = 182
    SELF_LENGTH = 61

    trainX, trainY = loader.dataset.get_io('2012-01-01', '2015-12-31')
    testX, testY = loader.dataset.get_io('2012-01-01', '2016-06-30')

    print(testX.shape)
    print(testY.shape)
    with torch.no_grad():
        train_period_input = to_gpu(trainX)
        train_period_output = to_gpu(trainY) * TOTAL_STD
        test_period_input = to_gpu(testX)
        test_period_output = to_gpu(testY[-TEST_LENGTH:]) * TOTAL_STD


    model = to_gpu(Model(input_dim=args.input_dim, hidden_size=args.hidden_size, output_dim=args.output_dim))

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

        test_out, hidden = model.forecast(test_period_input)
        test_period_forecast = test_out[-TEST_LENGTH:] * TOTAL_STD
        test_mse = criterion(test_period_forecast, test_period_output)**.5
        test_mae = metric(test_period_forecast, test_period_output)

        # rolling with cached seasonal series
        outputs = []
        output, h = model.forecast(torch.cat((to_gpu(testY[-1]).unsqueeze(0), cache[0].unsqueeze(0)), 1), hidden[-1])
        outputs += [output]
        i=1
        for _ in range(SELF_LENGTH):
            output, h = model.forecast(torch.cat((output, cache[i%7].unsqueeze(0)), 1), h[-1])
            outputs += [output]
            i+=1
        roll_output1 = torch.stack(outputs, 1).squeeze(2)[0] * TOTAL_STD
        # print(roll_output1.shape, self_test_output.shape)
        self_mse_cache = criterion(roll_output1, self_test_output)**.5
        self_mae_cache = metric(roll_output1, self_test_output)
        self_mse_cache_W = criterion(roll_output1[:14], self_test_output[:14])**.5
        self_mae_cache_W = metric(roll_output1[:14], self_test_output[:14])


        # rolling with seasonal decomposition
        outputs = []
        state = test_period_output1[-1]
        s = model1.forecast(to_gpu(testY[-1]).unsqueeze(0), state)
        output, h = model.forecast(torch.cat((to_gpu(testY[-1]).unsqueeze(0), s[2]), 1), hidden[-1])
        outputs += [output]
        for _ in range(SELF_LENGTH):
            s = model1.forecast(output, s[-1])
            output, h = model.forecast(torch.cat((output, s[2]), 1), h[-1])
            outputs += [output]
        roll_output2 = torch.stack(outputs, 1).squeeze(2)[0] * TOTAL_STD
        self_mse_decom = criterion(roll_output2, self_test_output)**.5
        self_mae_decom = metric(roll_output2, self_test_output)
        self_mse_decom_W = criterion(roll_output2[:14], self_test_output[:14])**.5
        self_mae_decom_W = metric(roll_output2[:14], self_test_output[:14])


        writer.add_scalar('train_mse', train_mse.data, global_step=epoch)
        writer.add_scalar('train_mae', train_mae.data, global_step=epoch)
        writer.add_scalar('test_mse', test_mse.data, global_step=epoch)
        writer.add_scalar('test_mae', test_mae.data, global_step=epoch)
        writer.add_scalar('self_mse_cache', self_mse_cache.data, global_step=epoch)
        writer.add_scalar('self_mae_cache', self_mae_cache.data, global_step=epoch)
        writer.add_scalar('self_mse_cache_W', self_mse_cache_W.data, global_step=epoch)
        writer.add_scalar('self_mae_cache_W', self_mae_cache_W.data, global_step=epoch)
        writer.add_scalar('self_mse_decom', self_mse_decom.data, global_step=epoch)
        writer.add_scalar('self_mae_decom', self_mae_decom.data, global_step=epoch)
        writer.add_scalar('self_mse_decom_W', self_mse_decom_W.data, global_step=epoch)
        writer.add_scalar('self_mae_decom_W', self_mae_decom_W.data, global_step=epoch)

        for _, (x, y, item) in enumerate(loader):
            x = to_gpu(x)
            y = to_gpu(y)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)
            writer.add_scalar('train_loss', loss.data, global_step=global_step)
            loss.backward(retain_graph=True)
            clip_grads(model)
            optimizer.step()
            global_step += 1

        torch.save(model.state_dict(),
                   os.path.join(writer_path, "snapshots%s.pth" % (epoch+1)))

    writer.export_scalars_to_json(os.path.join(writer_path, "history.json"))
