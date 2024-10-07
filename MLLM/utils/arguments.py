import argparse
def get_args():
    parser = argparse.ArgumentParser()
    # args for randomness
    parser.add_argument('--seed', type=int, default=2048, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', default=False, action='store_true', help='set cudnn.deterministic True')
    # args for data
    parser.add_argument('--train_data_jsons', type=str, nargs="+", help="list of train data jsons, separated by comma,")
    parser.add_argument('--valid_data_jsons', type=str, nargs="+", help="list of valid data jsons, separated by comma,")
    parser.add_argument('--batch_scale', type=int, default=1000, help="summed sequence length of each batch")
    parser.add_argument('--max_length', type=int, default=1000, help="maximum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--min_length', type=int, default=100, help="minimum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--n_worker', type=int, default=4, help='number of loading workers for each GPU')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=500, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--global_learning_rate', type=float, default=2e-6, help='The learning rate for training')
    parser.add_argument('--local_learning_rate', type=float, default=4e-6, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=10000, help="step of warmup")
    parser.add_argument('--total_steps', type=int, default=1000000, help="step of warmup")

    # args for model
    parser.add_argument('--dim', type=int, default=4096, help='the dimension of model')
    parser.add_argument('--text_card', type=int, default=1, help='the text token space of LLM')
    parser.add_argument('--existing_text_padding_id', type=int, default=3, help='The text_padding_id')
    parser.add_argument('--n_q', type=int, default=16, help='the two stream audio token space for MLLM')
    parser.add_argument('--dep_q', type=int, default=8, help="the depth of local transformer or depth transformer")
    parser.add_argument('--card', type=int, default=2048, help="the audio token space of each layer")
    parser.add_argument('--num_heads', type=int, default=32, help="the head number")
    parser.add_argument('--num_layers', type=int, default=32, help="")
    parser.add_argument('--hidden_scale', type=float, default=4.5, help="")
    parser.add_argument('--causal', type=bool, default=True, help="")
    parser.add_argument('--context', type=int, default=3000, help="")


    # args for save model and log: 
    parser.add_argument('--exp_dir', type=str, default='./log', help='directory of this experiment')
    parser.add_argument('--print_freq', type=int, default=5, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='save a checkpoint within an epoch')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')
    
    args = parser.parse_args()
    
    return args