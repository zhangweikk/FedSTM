from copy import deepcopy
import random
from client import Client
from utils import *
from server import Server
from image_synthesizer import Synthesizer
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
np.set_printoptions(precision=4, suppress=True)
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

channel_dict =  {
  "cifar10": 3,
  "cinic10": 3,
  "cifar100": 3,
  "eurosat": 3,
  "NWPU_RESISC45": 3,
  "mnist": 1,
  "fmnist": 1,
}
imsize_dict =  {
  "eurosat": (32,32),
  "NWPU_RESISC45":(32 ,32),
  "cifar10": (32, 32),
  "cinic10": (32, 32),
  "cifar100": (32, 32),
  "mnist": (28, 28),
  "fmnist": (28, 28),
}

hyperparameters01 = [{
    "random_seed": [4],
    "dataset": ["cifar10"],
    "models": [{"ConvNet": 80}],
    "attack_rate": [0],
    "attack_method": ["-"],
    "participation_rate": [0.4],
    "alpha": [0.01],
    "eta": [0.4],
    "client_mode": ["normal"],
    "minimum_trajectory_length": [[25,150]],
    "maximum_distill_round": [1],
    "distill_interval": [1],
    "start_round": [0],
    "communication_rounds": [400],
    "local_epochs": [1],
    "batch_size": [32],
    "val_size": [32],
    "val_batch_size": [32],
    "local_optimizer": [["Adam", {"lr": 0.001}]],
    "distill_iter": [20],
    "distill_lr": [1e-4],
    "aggregation_mode": ["datadistill"],
    "sample_size": [0],
    "save_scores": [False],  
    "pretrained": [None],   
    "save_model": [None],
    "log_frequency": [1],
    "log_path": ["new_noniid/"]
}]


hyperparameters_json = json.dumps(hyperparameters01)


import os
# os.environ["WANDB_MODE"]="offline"


parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--hp", default=hyperparameters_json, type=str)

parser.add_argument("--project", default='FedSTM', type=str)

parser.add_argument("--DATA_PATH", default="../data/", type=str)

parser.add_argument("--runs_name", default='finalcifar0.01ipc15and50threenumworker0', type=str)

parser.add_argument("--RESULTS_PATH", default="results/", type=str)

parser.add_argument("--ACC_PAT H", default=None, type=str)

parser.add_argument("--CHECKPOINT_PATH", default="checkpoints/", type=str)

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--ipc', type=int, default=15, help='image(s) per class')

parser.add_argument('--Iteration', type=int, default=4000, help='how many distillation steps to perform')

parser.add_argument('--lr_img', type=float, default=5e-2, help='learning rate for updating synthetic images')
parser.add_argument('--lr_label', type=float, default=1e-2, help='learning rate for updating synthetic images')

parser.add_argument('--least_ave_num', type=int, default=2, help='learning rate for updating synthetic images')

parser.add_argument('--max_ave_num', type=int, default=10, help='learning rate for updating synthetic images')

parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')

parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
parser.add_argument('--label_init', type=float, default=0, help='how to init lr (alpha)')

parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM   如果不是cifar100设置为none')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

parser.add_argument('--r', type=str, default='real', choices=["noise", "real"],
                    help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                    help='whether to use differentiable Siamese augmentation.')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')
parser.add_argument('--pix_init', type=str, default='noise',
                    help='differentiable Siamese augmentation strategy')


parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--img_optim', type=str, default='adam', help='dataset path')
parser.add_argument('--lr_optim', type=str, default='adam', help='dataset path')

parser.add_argument('--buffer_path', type=str, default=None, help='buffer path')
parser.add_argument('--expert_dir', type=str, default='./buffers', help='buffer path')
parser.add_argument('--start_learning_label', type=int, default=0, help='how many expert epochs the target params are')
parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
parser.add_argument('--max_start_epoch', type=int, default=50, help='max epoch we can start at')

parser.add_argument('--min_start_epoch', type=int, default=0, help='max epoch we can start at')

parser.add_argument('--max_epoch_incre', type=int, default=5, help='max epoch we can start at')
parser.add_argument('--classes', type=int, default=None, nargs='+', help='max epoch we can start at')
parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

parser.add_argument('--random_weights', action='store_true', help="will distill textures instead")
parser.add_argument('--weight_averaging', default=True, action='store_true', help="will distill textures instead")

parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')



#########################################################
parser.add_argument('--batch_syn2', type=int, default=64, help='should only use this if you run out of VRAM')
parser.add_argument('--ipc2', type=int, default=50, help='image(s) per class')
parser.add_argument('--min_start_epoch2', type=int, default=100, help='max epoch we can start at')
parser.add_argument('--max_start_epoch2', type=int, default=200, help='max epoch we can start at')


###########################################################
parser.add_argument('--use_feddecor',default = True, action='store_true', help="local update")
parser.add_argument('--feddecorr_coef', type=float, default=0.05,
                            help='coefficient of the FedDecorr loss')
###########################################################


args = parser.parse_args()

args.RESULTS_PATH = os.path.join(args.RESULTS_PATH, args.dataset, args.runs_name, str(random.randint(0,1000)))

if not os.path.exists(args.RESULTS_PATH):
  os.makedirs(args.RESULTS_PATH)

def run_experiment(xp, xp_count, n_experiments):
  t0 = time.time()
  print(xp)
  hp = xp.hyperparameters
  run = wandb.init(project = args.project, config = hp, reinit = True, name=args.runs_name)
  print(wandb.config)
  args.dsa = True
  args.dsa_param = ParamDiffAug()

  num_classes = {"NWPU_RESISC45": 45, "eurosat" : 10,"mnist" : 10, "fmnist" : 10, "cifar10" : 10,"cinic10" : 10, "cifar100" : 100, "nlp" : 4, 'news20': 20}[hp["dataset"]]
  # 10
  if hp.get("loader_mode", "normal") != "normal":
    num_classes = 3
  args.dsa = True
  args.dsa_param = ParamDiffAug()
  args.num_classes = num_classes
  
  args.channel = channel_dict[hp['dataset']]
  
  args.imsize = imsize_dict[hp['dataset']]

  if args.batch_syn is None:
    args.batch_syn = num_classes * args.ipc

  print(f"num classes {num_classes}, dsa mode {hp.get('dsa', True)}")
  model_names = [model_name for model_name, k in hp["models"].items() for _ in range(k)]
  
  optimizer, optimizer_hp = getattr(torch.optim, hp["local_optimizer"][0]), hp["local_optimizer"][1]

  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})
  print(f"dataset : {hp['dataset']}")

  train_data_all, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  np.random.seed(hp["random_seed"])
  torch.manual_seed(hp["random_seed"])

  '''
  load dataset
  '''

  train_data = train_data_all
  
  if hp.get("loader_mode", "normal") == "normal":
    
    client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=len(model_names),
        alpha=hp["alpha"], batch_size=hp["batch_size"], n_data=None, num_workers=0, seed=hp["random_seed"])
  else:
    
    indices = torch.load("checkpoints/cifar10/ConvNet/0.01/823/sampled_indices.pth")
    client_loaders, test_loader, class_indices = data.get_loaders_classes(train_data, test_data, n_clients=len(model_names),
                                                   alpha=hp["alpha"], batch_size=hp["batch_size"], n_data=None, num_workers=0, seed=hp["random_seed"], classes =  [6,7,9], total_num = 6000, indices=indices)
  images_train, labels_train = None, None




  server = Server(np.unique(model_names), test_loader,test_loader,num_classes=num_classes, imsize=args.imsize, images_train=images_train, labels_train=labels_train, eta=hp.get('eta', 0) , dataset = hp['dataset'])
  clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, imsize=args.imsize,images_train=images_train, labels_train=labels_train, eta=hp.get('eta', 0), dataset = hp['dataset']) for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
  print(clients[0].model)




  # initialize data synthesizer
  synthesizer = Synthesizer(deepcopy(clients[0].model), test_loader, args)
  server.number_client_all = len(client_loaders)

  models.print_model(clients[0].model)

  print("Start Distributed Training..\n")
  t1 = time.time()
  xp.log({"prep_time" : t1-t0})




  
  maximum_acc_test, maximum_acc_val = 0, 0

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})

  test_accs, val_accs = [], []
  trajectories_list = []

  distilled_rounds = 0
  trajectories_list.append([])

  trajectories_list[-1].append([p.cpu() for p in server.model_dict[list(server.model_dict.keys())[0]].parameters()])
  print(f"model key {list(server.model_dict.keys())[0]}")

  '''
   start training
   '''



  for c_round in range(1, hp["communication_rounds"]+1):


##################################################################
        # First Synthesis
    if len(trajectories_list[distilled_rounds]) == hp["minimum_trajectory_length"][0]:
      print(f"{c_round + 1}th iteration, update synthesized data ...")

      synthesizer.synthesize(trajectories_list=trajectories_list, args=args)
      synthesizer.evaluate(c_round + 1, args=args)
      server.images_train, server.labels_train = synthesizer.image_syn.cpu().detach(), synthesizer.label_syn.cpu().detach()
        # Second Synthesis
      kkkk = hp["minimum_trajectory_length"][1]
    if len(trajectories_list[distilled_rounds]) == hp["minimum_trajectory_length"][1]:
      print('开始第二次合成')
      print(f"{c_round+1}th iteration, update synthesized data ...")

      synthesizer.synthesize2(trajectories_list=trajectories_list, args=args)
      synthesizer.evaluate2(c_round+1, args=args)
      server.images_train, server.labels_train =  synthesizer.image_syn2.cpu().detach(), synthesizer.label_syn2.cpu().detach()

#################################################################################







    
    participating_clients = server.select_clients(clients, hp["participation_rate"], hp.get('unbalance_rate', 1), hp.get('sample_mode', "uniform"))
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})
    client_avg_loss = []
    for client in participating_clients:
        client.synchronize_with_server(server)
        train_stats = client.compute_weight_update(hp["local_epochs"], lambda_fedprox=hp["lambda_fedprox"] if "PROX" in hp["aggregation_mode"] else 0.0, args=args)
        if 'loss' in train_stats:
            client_avg_loss.append(train_stats['loss'])
   
    avg_loss = sum(client_avg_loss) / len(client_avg_loss)
    print(f"所有客户端的平均损失为: {avg_loss:.6f}")
    if hp["aggregation_mode"] == "FedAVG":
      server.fedavg(participating_clients)
    elif hp["aggregation_mode"] == "ABAVG":
      server.abavg(participating_clients)
    elif hp["aggregation_mode"] == "datadistill":
    
      distill_iter = hp.get("distill_iter", None)
     
      distill_lr = hp.get("distill_lr", None)
      server.datadistill(participating_clients, distill_iter, distill_lr, dsa=hp.get("dsa", True), args=args)
    elif "PROX" in hp["aggregation_mode"]:
      server.fedavg(participating_clients)
    else:
      import pdb; pdb.set_trace()

    if xp.is_log_round(c_round):
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      wandb.log({"client_avg_loss_a_{}".format(hp["alpha"]): avg_loss}, step=c_round)
      if server.weights != None:
        xp.log({"weights": np.array(server.weights.cpu())})

      for key, value in server.evaluate_ensemble().items():
        if key == "test_accuracy":
          if value > maximum_acc_test:
            maximum_acc_test = value
            wandb.log({"maximum_acc_{}_a_{}_test".format("accuracy", hp["alpha"]): maximum_acc_test}, step=c_round)
        elif key == "val_accuracy":
          if value > maximum_acc_val:
            maximum_acc_val = value
            wandb.log({"maximum_acc_{}_a_{}_val".format("accuracy", hp["alpha"]): maximum_acc_val}, step=c_round)
      
      xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})
      wandb.log({"server_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_ensemble().items()}, step=c_round)
      xp.log({"epoch_time" : (time.time()-t1)/c_round})
      stats = server.evaluate_ensemble()
      test_accs.append(stats['test_accuracy'])
      val_accs.append(stats['val_accuracy'])
      # Save results to Disk
      xp.save_to_disc(path=args.RESULTS_PATH, name="logfiles")
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))
    trajectories_list[-1].append([p.cpu() for p in server.model_dict[list(server.model_dict.keys())[0]].parameters()])

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()
  run.finish()

def run():
  experiments_raw = json.loads(args.hp)
  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))
 
  
if __name__ == "__main__":
  # import wandb
  import swanlab as wandb
  
  run()
   