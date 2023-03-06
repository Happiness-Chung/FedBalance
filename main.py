'''
Main file to set up the FL system and train
Code design inspired by https://github.com/mmendiet/FedAlign
'''
import torch
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_fedalign import resnet56 as resnet56_fedalign
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time

# methods
import methods.fedavg as fedavg
import methods.fedprox as fedprox
import methods.moon as moon
import methods.fedalign as fedalign
import methods.fedbalance as fedbalance
import data_preprocessing.custom_multiprocess as cm

def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedbalance', metavar='N',
                        help='Options are: fedavg, fedprox, moon, fedalign, fedbalance')
    
    parser.add_argument('--harmony', type=str, default='n', metavar='N')
    
    parser.add_argument('--data_dir', type=str, default="C:/Users/hb/Desktop/data/NIH",
                        help='data directory: data/cifar100, data/cifar10, "C:/Users/hb/Desktop/data/NIH", C:/Users/hb/Desktop/data/CheXpert-v1.0-small')

    parser.add_argument('--dataset', type=str, default="NIH",
                        help='data directory: cifar100, cifar10, NIH, CheXpert')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=5, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=30,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,  
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=1.0, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--width', type=float, default=0.25, metavar='WI',
                        help='minimum width for subnet training')

    parser.add_argument('--mult', type=float, default=0.0001, metavar='MT',
                        help='multiplier for subnet training')

    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets sampled during training')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=1, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')

    parser.add_argument('--gamma', default=0.0, type=float,
                    help='hyperparameter gamma for mixup')
    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    # q is the client info
    set_random_seed()
    global client 
    # c0 is a client_dict
    # c1 is the namespace
    ci = q.get() 
    client = Client(ci[0], ci[1]) 

def run_clients(received_info):
    try:
        return client.run(received_info) # give threads' number of model weight
    except KeyboardInterrupt:
        logging.info('exiting')
        return None

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list) 
    for round in range(args.comm_round): 
        if args.client_sample<1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else: 
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number) 
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)] 
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict

if __name__ == "__main__":
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
 
    ###################################### get data
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num, client_pos_freq, client_neg_freq, client_imbalances = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size)

    mapping_dict = allocate_clients_to_threads(args) 
    print("Client allocation for the threads during commication round : ", mapping_dict)
    if args.method=='fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir} for i in range(args.thread_number)]
    elif args.method=='fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony, 'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='moon':
        Server = moon.Server
        Client = moon.Client
        Model = resnet56 
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='fedalign':
        Server = fedalign.Server
        Client = fedalign.Client
        Model = resnet56_fedalign 
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'harmony': args.harmony,'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions, 'dir': args.data_dir, 'harmony': args.harmony,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    elif args.method=='fedbalance':
        Server = fedbalance.Server
        Client = fedbalance.Client
        Model = resnet56
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'dir': args.data_dir, 'imbalances': client_imbalances}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions, 'dir': args.data_dir,
                            'clients_pos': client_pos_freq, 'clients_neg': client_neg_freq} for i in range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')
    
    #init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))
    # the length of the client info is the number of threads

    ######################################################
    # Start server and get initial outputs
    pool = cm.MyPool(args.thread_number, init_process, (client_info, Client)) 
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}__{}_e{}_c{}'.format(os.getcwd(), args.dataset, time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs, args.client_number)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    server = Server(server_dict, args) # Server initializaion
    # methods.fedavg.Server object
    server_outputs = server.start() 
    # weight of the server
    # Start Federated Training
    # the length is the number of treads
    time.sleep(150*(args.client_number/16)) #  Allow time for threads to start up
    for r in range(args.comm_round):
        logging.info('***** Round: {} ************************'.format(r))
        round_start = time.time()
        # server output length :        
        client_outputs = pool.map(run_clients, server_outputs) 
        client_outputs = [c for sublist in client_outputs for c in sublist]  
        # sublist : 'weights': OrderedDict
        # length : the number of clients
        # c is the weight of a client   
        server_outputs = server.run(client_outputs) 
        round_end = time.time()
        total_sec = round_end-round_start
        total_min = (total_sec) // 60
        logging.info('Round {} Time: {:.0f}m {:.0f}s'.format(r, total_min, total_sec % 60))
    pool.close()
    pool.join()
