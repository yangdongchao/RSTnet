import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--training_dtype', type=int, default=-1)
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=500, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--global_learning_rate', type=float, default=2e-6, help='The learning rate for training')
    parser.add_argument('--local_learning_rate', type=float, default=4e-6, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=10000, help="step of warmup")
    parser.add_argument('--total_steps', type=int, default=1000000, help="total training step")

    # args for local model
    parser.add_argument('--audio_card', type=int, default=2050, help='the text token space of LLM')
    parser.add_argument('--codecformer_dim', type=int, default=1024, help='The dimension of codecformer')
    parser.add_argument('--n_q', type=int, default=8, help='the two stream audio token space for MLLM')
    parser.add_argument('--dep_q', type=int, default=8, help="the depth of local transformer or depth transformer")
    parser.add_argument('--codecformer_heads', type=int, default=16, help="the head number")
    parser.add_argument('--codecformer_layers', type=int, default=6, help="")
    parser.add_argument('--codecformer_hidden_scale', type=float, default=4.5, help="")
    parser.add_argument('--causal', type=str2bool, default=True, help="")
    parser.add_argument('--codecformer_multi_linear', type=str2bool, default=True, help="whether use multiple linear layer")
    parser.add_argument('--codecformer_weights_per_step', type=str2bool, default=True, help="different weight for codecformer")
    parser.add_argument('--codecformer_dim_feedforward', type=int, default=4224, help="")
    parser.add_argument('--codecfomer_norm', type=str, default='rms_norm_f32', help="the layer norm method for the codecformer")
    parser.add_argument('--codecformer_bias_proj', type=str2bool, default=False, help="different weight for codecformer")
    parser.add_argument('--codecfomer_norm_emb', type=bool, default=False)


    # args for lora config
    parser.add_argument('--lora_r', type=int, default=8, help='The LoRA rank.')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='The LoRA dropout value.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='The LoRA dropout value.')
    parser.add_argument('--lora_query', type=str2bool, default=True, help='Whether to apply LoRA to the query weights in attention.')
    parser.add_argument('--lora_key', type=str2bool, default=False, help='Whether to apply LoRA to the key weights in attention.')
    parser.add_argument('--lora_value', type=str2bool, default=True, help='Whether to apply LoRA to the value weights in attention.')
    parser.add_argument('--lora_projection', type=str2bool, default=False, help='Whether to apply LoRA to the output projection in the attention block.')
    parser.add_argument('--lora_mlp', type=str2bool, default=False, help='Whether to apply LoRA to the weights of the MLP in the attention block.')
    parser.add_argument('--lora_head', type=str2bool, default=False, help='Whether to apply LoRA to output head in GPT.')


    # args for save model and log: 
    parser.add_argument('--parallel_number', type=int, default=9, help='the number of training streaming')
    parser.add_argument('--exp_dir', type=str, default='./log', help='directory of this experiment')
    parser.add_argument('--model_config', type=str, default='configs/llama3.yaml', help='the config file for LLM')
    parser.add_argument('--checkpoint_path', type=str, default='/weka2/home-dongchao/data/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/lit_model.pth')
    parser.add_argument('--print_freq', type=int, default=100, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='save a checkpoint within an epoch')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')

    # dataloader config
    parser.add_argument('--audio_tokenizer', type=str, default='mimi', help='the type of audio tokenizer')
    parser.add_argument('--text_tokenizer', type=str, default='llama3-8B', help='the type of audio tokenizer')
    args = parser.parse_args()
    
    return args