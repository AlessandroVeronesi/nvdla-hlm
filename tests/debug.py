
import os
import sys
import argparse
import yaml

import numpy as np
import torch
from random import randrange, randint
from datetime import datetime

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'src'))

import nvdla

##############################

MAXHEIGHT    = 10
MAXWIDTH     = 10
MAXWHEIGHT   = 7
MAXWWIDTH    = 7
MAXSTRIDE    = 5
MAXPADDING   = 5
MAXDILATION  = 1 # FIXME: Dilation > 1 Currently not supported

##############################

def randval(input):
    return randrange(input) if (input > 0) else 0

def cfgCheck(configfile):

    if not os.path.isfile(dbgcfg):
        print(f'Configuration File {dbgcfg} does not exists')
        return False

    with open(configfile, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    if(config['dat-type'] == config['wt-type']):
        return True
    else:
        print(f'Configuration File {dbgcfg} has unmatching wt and dat data types (feature not supported)')
        return False

def conv(ftens, ktens, bias, stride, padding, dilation):
    if((ftens.ndim != 4) or (ktens.ndim != 4)):
        raise Exception('Invalid input dimensions')
    
    B, C, H, W = ftens.shape
    K,C_, R, S = ktens.shape

    if(C != C_):
        raise Exception('Invalid input dimensions')

    ftens_torch = torch.from_numpy(ftens)
    ktens_torch = torch.from_numpy(ktens)
    bias_torch  = torch.from_numpy(bias)

    otens_torch = torch.nn.functional.conv2d(ftens_torch, ktens_torch, bias=bias_torch, stride=stride, padding=padding, dilation=dilation)

    return otens_torch.numpy()

##############################

def testroutine(nvdla, dbgcfg, testnum, verbose=False):

    MAXB = -1
    MAXC = -1
    MAXK = -1
    with open(dbgcfg, 'r') as f:
        cfgdict = yaml.load(f, yaml.SafeLoader)
        # MAXB    =      cfgdict['cmac']['batch-size']
        # MAXC    = 4 *  cfgdict['cmac']['atomic-c']
        # MAXK    = 32 * cfgdict['cmac']['atomic-k']
        MAXB    = cfgdict['cmac']['batch-size']
        MAXC    = cfgdict['cmac']['atomic-c']
        MAXK    = cfgdict['cmac']['atomic-k']


    for testit in range(testnum):

        input_gen_exit = False
        while(not input_gen_exit):
            stride   = ((randval(MAXSTRIDE-1)   +1), (randval(MAXSTRIDE-1)   +1))
            padding  = ((randval(MAXPADDING)),       (randval(MAXPADDING))      )
            dilation = ((randval(MAXDILATION-1) +1), (randval(MAXDILATION-1) +1))
            B = randval(MAXB-1) +1
            K = randval(MAXK-1) +1
            C = randval(MAXC-1) +1
            R = randval(MAXWHEIGHT-1) +1
            S = randval(MAXWWIDTH-1)  +1

            R_ = (R-1)*dilation[0] +1
            S_ = (S-1)*dilation[1] +1

            H_ = randval(MAXHEIGHT) +1
            W_ = randval(MAXWIDTH) +1

            H = (H_-1)*stride[0] -2*padding[0] +R_
            W = (W_-1)*stride[1] -2*padding[1] +S_

            ##################

            stride = (1,1)
            padding = (0,0)
            dilation = (1,1)

            B = 1
            K = randval(MAXK-1) +1
            C = randval(MAXC-1) +1

            R = 2
            S = 2

            H = 3
            W = 3

            ##################

            if((H_>0) and (W_>0) and (H>0) and (W>0) and (R_>0) and (S_>0)):
                input_gen_exit = True

        Fmap = np.random.uniform(-10,10, (B,C,H,W)).astype(np.float32)
        Kmap = np.random.uniform(-10,10, (K,C,R,S)).astype(np.float32)
        Bias = np.random.uniform(-10,10, K).astype(np.float32)

        print('-I: Running Test with:')
        print(f'-I: Fmap     = {Fmap.shape}')
        print(f'-I: Kmap     = {Kmap.shape}')
        print(f'-I: Bias     = {Bias.shape}')
        print(f'-I: Stride   = {stride}')
        print(f'-I: Padding  = {padding}')
        print(f'-I: Dilation = {dilation}')

        # input('...')

        if verbose:
            print('-I: inputs tensor is:')
            print(Fmap)
            print('-I: weigths tensor is:')
            print(Kmap)
            print('-I: Bias array is:')
            print(Bias)

        golden = conv(Fmap, Kmap, Bias, stride, padding, dilation)

        start  = datetime.now()
        dut    = nvdla.convolve(Fmap, Kmap, Bias, stride, padding, dilation)
        stop   = datetime.now()

        if((golden.shape == dut.shape) and np.allclose(golden, dut, rtol=args.threshold)):
            print(f'-I: Test {testit} PASSED (elapsed: {(stop-start).total_seconds()*1000} ms)')
            if verbose:
                print('-I: output tensor is:')
                print(dut)
        else:
            if(golden.shape != dut.shape):
                print(f'-E: Expected tensor of shape {golden.shape}, but received of shape {dut.shape}')
                print(f'-E: Test {testit} FAILED')
                sys.exit('-E: DEBUG  FAILED')
            else:
                (B,C,H,W) = golden.shape
                for b in range(B):
                    for c in range(C):
                        if not ((golden[b][c].shape == dut[b][c].shape) and np.allclose(golden[b][c], dut[b][c], rtol=args.threshold)):
                            print(f'-E: First error in sample {b} at channel {c}')
                            print('--------------------------------------------------')
                            print('Golden:')
                            print(golden[0][0])
                            print('...')
                            print(golden[b][c])
                            print('--------------------------------------------------')
                            print('DUT')
                            print(dut[0][0])
                            print('...')
                            print(dut[b][c])
                            print('--------------------------------------------------')
                            print(f'Features Tensor Parameters: {Fmap.shape}')
                            print(f'Weights Tensor Parameters:  {Kmap.shape}')
                            print(f'Golden Tensor Parameters:   {golden.shape}')
                            print(f'DUT Tensor Parameters:      {dut.shape}')
                            print('--------------------------------------------------')
                            print(f'-E: Test {testit} FAILED')
                            sys.exit('-E: DEBUG  FAILED')

##############################

if __name__ == '__main__':

    ##############################
    # ARGS

    dftcfg = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'specs', 'debug.yaml')

    parser = argparse.ArgumentParser()

    parser.add_argument('--testnum', type=int, default=1, help='Set number of tests')
    parser.add_argument('--threshold', type=float, default=1e-1, help='Set maximum error tolerance in output check')
    parser.add_argument('--verbose', action='store_true', help='Verbose Testing')
    parser.add_argument('--config', default=dftcfg, help='Config identifier file')

    args = parser.parse_args()

    ##############################
    # INFOs

    pid = os.getpid()

    dbgcfg = os.path.join(os.path.dirname(__file__), '..', 'specs', 'debug16_int8.yaml')

    if args.config is not None:
        dbgcfg = args.config

    if not cfgCheck(dbgcfg):
        sys.exit('-E: DEBUG  FAILED')

    print(f'\n-I: NVDLASIM Debug')
    print(f'-I: PID = {pid}')
    print(f'-I: Debug Config = {dbgcfg}')
    print(f'-I: Running {args.testnum} tests...')

    ##############################
    # Init NVDLA

    nvdla = nvdla.core(dbgcfg)

    ##############################
    # Testing

    testroutine(nvdla, args.config, args.testnum, args.verbose)

    print('-I: ALL TESTS PASSED')
