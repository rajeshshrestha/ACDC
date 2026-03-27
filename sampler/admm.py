import torch
import wandb as wnb
import numpy as np
import torch.nn as nn
from utils.diffusion import Scheduler, DiffusionSampler
import tqdm
from utils.logging import Trajectory
from .registry import register_sampler


@register_sampler(name='admm')
def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError
    return ADMM(**kwargs)


class ADMM(nn.Module):
    """Implemenation of ADMM based MAP reconstruction."""

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config,
                 lgvd_config, admm_config, device='cuda', **kwargs):
        super().__init__()
        self.annealing_scheduler_config, self.diffusion_scheduler_config = \
            self._check(annealing_scheduler_config, diffusion_scheduler_config)

        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        
        self.admm_config = admm_config
        self.device = device

        '''Initialize the diffusion parameters'''
        self.betas = np.linspace(admm_config.denoise.diffusion.beta_start,
                                 admm_config.denoise.diffusion.beta_end,
                                 admm_config.denoise.diffusion.T,
                                 dtype=np.float64)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        

        self.regularizers = None


    def optimize_ml_with_generic_gd(self, x_k, z_k, u_k, operator, measurement, wandb=False):
        ml_config = self.admm_config.ml

        ml_loss_lst = []
        progress_bar = tqdm.trange(ml_config.max_iter) if ml_config.verbose \
            else range(ml_config.max_iter)
        last_loss = np.inf

        '''Optimizer'''
        x_k.requires_grad = True
        lr = ml_config.lr
        if ml_config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam([x_k], lr=lr)
        elif ml_config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD([x_k], lr=lr)
        else:
            raise ValueError(f"Optimizer {ml_config.optimizer} not supported")

        conv_count = 0
        for iteration in progress_bar:

            '''Optimize the loss function'''
            optimizer.zero_grad()
            lk_loss = operator.loss(x_k, measurement)
            reg_loss = self.admm_config.rho/2 * ((x_k - z_k + u_k)**2).sum()
            loss_val = reg_loss + lk_loss

            if ((('reg_use_freq' in ml_config) and (iteration % ml_config.reg_use_freq == 0)) or 'reg_use_freq' not in ml_config) and self.regularizers:
                extra_reg_loss = sum([reg(x_k) for reg in self.regularizers])
                loss_val += extra_reg_loss
            else:
                extra_reg_loss = torch.tensor(0).to(self.device)

            loss_val.backward()
            optimizer.step()

            '''Run post ml operation if any'''
            x_k = operator.post_ml_op(x_k, measurement)

            '''Clipping x_k'''
            if ml_config.clip:
                with torch.no_grad():
                    x_k.clamp_(-1.0, 1.0)

            delta_loss = abs(last_loss - loss_val.item())
            ml_loss_lst.append(loss_val.item())
            if ml_config.verbose:
                progress_bar.set_description(
                    f"Lr: {lr:.6f} Rho: {self.admm_config.rho: .6f} ML Loss: {loss_val.item():.2f} " +
                    f"Lk Loss: {lk_loss.item():.2f} " +
                    f"Reg Loss: {reg_loss.item():.2f} " +
                    f"Extra Reg Loss: {extra_reg_loss.item():.2f} " +
                    f"Delta Loss: {delta_loss:.2f}")

            '''Adaptive learning rate and check for convergence'''
            if last_loss < loss_val.item():
                lr /= ml_config.lr_decay
                if lr < ml_config.lr_min:
                    break
                '''Update learning rate'''
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            elif delta_loss < ml_config.tol:
                conv_count += 1
                if conv_count > ml_config.patience:
                    break
            else:
                conv_count = 0
            last_loss = loss_val.item()

        return x_k.detach()

    def optimize_ml(self, x_k, z_k, u_k, operator, measurement, wandb=False):
        if "use_task_specific_solver" in self.admm_config.ml and self.admm_config.ml.use_task_specific_solver.activate:
            print("Using task specific solver")
            return operator.ml_solver(
                x_k=x_k,
                z_k=z_k,
                u_k=u_k,
                rho=self.admm_config.rho,
                measurement=measurement,
                solver_config=self.admm_config.ml.use_task_specific_solver,
                wandb=wandb)
        else:
            return self.optimize_ml_with_generic_gd(
                x_k=x_k, z_k=z_k, u_k=u_k,
                operator=operator,
                measurement=measurement,
                wandb=wandb)

    def optimize_denoising(self, x_k, u_k,
                           model, sigma,
                           prior_use_type="denoise",
                           wandb=False):

        denoise_config = self.admm_config.denoise
        with torch.no_grad():
            '''Compute the noisy image'''
            noisy_im = (x_k + u_k).clone()

            if prior_use_type in ["denoise"]:
                '''Approximate correction'''
                forward_z = noisy_im + torch.randn_like(noisy_im) * sigma

                '''Dirrectional Correction using lgvd'''
                lgvd_z = forward_z.clone()
                lr = denoise_config.lgvd.lr * sigma
                for _ in range(denoise_config.lgvd.num_steps):
                    score_val = model.score(lgvd_z, sigma)
                    diff_val = (forward_z - lgvd_z)
                    lgvd_z += lr * score_val +\
                        lr * min(sigma * denoise_config.lgvd.reg_factor, 10) * diff_val +\
                        (2*lr)**0.5 * torch.randn_like(noisy_im)

                if denoise_config.final_step == 'tweedie':
                    z = model.tweedie(lgvd_z, sigma)

                elif denoise_config.final_step == 'ode':
                    diffusion_scheduler = Scheduler(
                        **self.diffusion_scheduler_config, sigma_max=sigma)
                    sampler = DiffusionSampler(diffusion_scheduler)
                    z = sampler.sample(model, lgvd_z, SDE=False, verbose=False)
                else:
                    raise Exception(
                        f"Final step {denoise_config.final_step} not supported!!!")

                denoised_img = torch.clamp(z, min=-1.0, max=1.0)
            else:
                raise Exception(
                    f"Prior type {prior_use_type} not supported!!!")
        
        return denoised_img

    def sample(self, model, ref_img, operator,
               measurement, evaluator=None,
               record=False, verbose=False, wandb=False, **kwargs):
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.admm_config.max_iter) if verbose \
            else range(self.admm_config.max_iter)

        '''Initialize x_k, z_k and u_k'''
        x_k, z_k, u_k = self.get_start(ref_img)

        '''Initialize x_k_old_vals'''
        x_k_old, z_k_old, u_k_old = None, None, None
        t_val = None

        '''Adaptive rho'''
        eta, gamma = self.admm_config.eta, self.admm_config.gamma
        delta_t_old = torch.inf

        delta_patience = 0
        for step in pbar:
            t_sigma = min(step, self.annealing_scheduler.num_steps-1)
            sigma = self.annealing_scheduler.sigma_steps[t_sigma]

            '''Optimize the ml subproblem'''
            x_k = self.optimize_ml(x_k=x_k, z_k=z_k, u_k=u_k,
                                   operator=operator,
                                   measurement=measurement,
                                   wandb=wandb)

            '''Optimize denoising subproblem'''
            z_k = self.optimize_denoising(
                x_k=x_k, u_k=u_k,
                model=model,
                sigma=sigma,
                prior_use_type=self.admm_config.denoise.type,
                wandb=wandb)

            '''Optimize the dual variable'''
            u_k = u_k + x_k - z_k

            if step != 0:
                delta_1 = 1/(256*256*3)*(x_k - x_k_old).norm()**2
                delta_2 = 1/(256*256*3)*(z_k - z_k_old).norm()**2
                delta_3 = 1/(256*256*3)*(u_k - u_k_old).norm()**2
                delta_t = delta_1 + delta_2 + delta_3

                '''Check for change convergence'''
                if delta_t < self.admm_config.delta_tol:
                    delta_patience += 1
                    if delta_patience > self.admm_config.delta_patience:
                        print(f"Converged with low delta at step {step}")
                        break
                if (delta_t > eta * delta_t_old) and (step > 0.8*self.annealing_scheduler.num_steps):
                    self.admm_config.rho *= gamma
                    self.admm_config.rho = min(self.admm_config.rho, 500)
                    u_k /= gamma
                delta_t_old = delta_t

                if wandb:
                    wnb.log({"ADMM Iteration": step+1, "delta_t": delta_t})

            '''Update the old values'''
            x_k_old, z_k_old, u_k_old = x_k.clone(), z_k.clone(), u_k.clone()

            # 4. evaluation
            x_k_results = z_k_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x_k_results = evaluator(gt, measurement, x_k)
                    z_k_results = evaluator(gt, measurement, z_k)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x_k' + '_' + main_eval_fn_name: f"{x_k_results[main_eval_fn_name].item():.2f}",
                        'z_k' + '_' + main_eval_fn_name: f"{z_k_results[main_eval_fn_name].item():.2f}",
                    })

                if wandb:
                    for fn_name in x_k_results.keys():
                        wnb.log({f'x_k_{fn_name}': x_k_results[fn_name].item(
                        ), f'z_k_{fn_name}': z_k_results[fn_name].item()})

            if record:
                self._record(u_k=u_k, x_k=x_k, z_k=z_k,
                             sigma=(1-self.alphas_cumprod[t_val])**0.5,
                             x_k_results=x_k_results,
                             z_k_results=z_k_results)
        return z_k

    # Code adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/sampler.py
    # Original author: bingliang
    def _record(self, u_k, x_k, z_k, sigma, x_k_results, z_k_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'x_k', x_k)
        self.trajectory.add_tensor(f'z_k', z_k)
        self.trajectory.add_tensor(f'u_k', u_k)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x_k_results.keys():
            self.trajectory.add_value(f'x_k_{name}', x_k_results[name])
        for name in z_k_results.keys():
            self.trajectory.add_value(f'z_k_{name}', z_k_results[name])
            

    # Code adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/sampler.py
    # Original author: bingliang
    def get_start(self, ref):
        init_values = []
        
        '''Initialize x_k, z_k and u_k'''
        for factor_key in self.admm_config.init_factor:
            if self.admm_config.init_factor[factor_key] is None:
                start_val = torch.randn_like(
                    ref) * self.annealing_scheduler.sigma_max
            else:
                start_val = torch.randn_like(
                    ref) * self.admm_config.init_factor.x

            init_values.append(start_val.to(self.device))

        return init_values
    

    # Code adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/sampler.py
    # Original author: bingliang
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config
