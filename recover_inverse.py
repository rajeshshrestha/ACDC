import hydra
from datasets import get_dataset
from measurements import get_operator
from setproctitle import setproctitle
from utils import set_seed
import torch
from omegaconf import OmegaConf
from model import get_model
from sampler import get_sampler
from utils.inverse_sampler import sample_in_batch
from utils.eval import Evaluator, get_eval_fn
from utils.logging import log_results
import wandb
from torch.utils.data import DataLoader



@hydra.main(version_base="1.3", config_path="configs", config_name="default_ffhq.yaml")
def main(args):
    print("============================================================")
    print("Running the inverse task")
    print("============================================================")
    if args.show_config:
        print(OmegaConf.to_yaml(args))
        print("\n")

    '''Set the seed, gpu and process name'''
    set_seed(args.seed)
    torch.cuda.set_device(f'cuda:{args.gpu}')

    '''Init wandb if required'''
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    '''Get the dataset'''
    dataset = get_dataset(**args.data)

    '''Get the forward measurement operator'''
    operator = get_operator(**args.inverse_task.operator)
    
    '''Update the annealing steps with task specific'''
    if 'annealing_steps' in args.inverse_task.admm_config.denoise:
        args.sampler.annealing_scheduler_config.num_steps = int(args.inverse_task.admm_config.denoise['annealing_steps'])
        args.inverse_task.admm_config.max_iter = args.sampler.annealing_scheduler_config.num_steps + 20
        print(f"Updated annealing_steps: {args.sampler.annealing_scheduler_config}")
        
    '''Get image from the dataset'''
    total_number = min(args.batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.total_images, shuffle=False)
    images = next(iter(dataloader)).to(f'cuda:{args.gpu}')
    y = operator.measure(images)

    '''Load the model'''
    model = get_model(**args.model)

    '''Load daps/admm sampler'''
    sampler = get_sampler(
        **args.sampler,
          **args.inverse_task)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # main sampling process
    full_samples = []
    full_trajs = []
    for r in range(args.num_runs):
        print(f'Run: {r}')
        samples, trajs = sample_in_batch(sampler, model, images, operator, y, evaluator, verbose=True,
                                         record=args.save_traj, batch_size=args.batch_size, gt=images, wandb=args.wandb)
        full_samples.append(samples)
        full_trajs.append(trajs)
    full_samples = torch.stack(full_samples, dim=0)

    # log metrics
    results = evaluator.report(images, y, full_samples)
    markdown_text = evaluator.display(results)
    print(markdown_text)

    # log results
    log_results(args, full_trajs, results, images, y,
                full_samples, markdown_text, min(args.total_images, len(dataset)))
    if args.wandb:
        evaluator.log_wandb(results, args.batch_size)
        wandb.finish()
    print(f"Finish the inverse tasks {args.name}")


if __name__ == "__main__":
    main()
