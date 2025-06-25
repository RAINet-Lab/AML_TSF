import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from models import DLinear

MODEL_MAP = {
    'dlinear': DLinear,
}

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

############ utilities ############

# Retrieve the flatten dataset ([dim0,dim1] => [dim0 + dim1])
def flatten_dataset(X):
        X_rs = X.reshape(X.shape[0],X.shape[1])
        if isinstance(X, torch.Tensor):
            result = torch.cat([X_rs[0], X_rs[1:, -1]], dim=0).squeeze()
        else:
            result = np.concatenate([X_rs[0], X_rs[1:, -1]]).squeeze()
        return result

# Opposite of the previous function ([dim0 + dim1] => [dim0,dim1])
def reshape_dataset(dataset, ref_dataset):
    dim0=ref_dataset.shape[0]
    dim1=ref_dataset.shape[1]
    tensor = torch.zeros(dim0,dim1)
    for i in range(dim0):
       tensor[i] = dataset[i:i+dim1]

    return tensor.unsqueeze(-1)

# Inject Traffic
def inject_traffic(dataset, points_to_poison, length, epsilon):
    adversarial_dataset = dataset.clone()
    for point in points_to_poison:
        for j in range(0,max(1,length)):
            if point+j<len(adversarial_dataset):
                # adversarial_dataset[point+j]+=abs(epsilon*adversarial_dataset[point+j])
                adversarial_dataset[point+j]+=epsilon
    return adversarial_dataset

# Get the actual contribution of each points
def sum_tensor_contribution_per_point(shap_tensor,dim0,dim1):
    shap_sum_of_each_point = torch.zeros(dim0+dim1)
    for i in range(dim0):
        shap_sum_of_each_point[i:i+dim1] += shap_tensor[i].abs().sum(dim=1)
    return shap_sum_of_each_point+1

# Normalize the tensor by the total contribution of each timestamp
def normalize_tensor(tensor):
    dim0=tensor.shape[0]
    dim1=tensor.shape[1]
    tensor_sum_of_each_point = sum_tensor_contribution_per_point(tensor.cpu(),dim0,dim1)
    tensor_normalized = torch.zeros(size=tensor.shape)
    for i in range(dim0):
        for j in range(dim1):
            # #print(shap_tensor_reduced.shape,normalize_coef.shape)
            tensor_normalized[i,j,:] = tensor[i,j,:]/(tensor_sum_of_each_point[i+j])

    return tensor_normalized

# Get the distribution of the sum the constribution per point
def distrib(matrix):
    sum_dim1 = matrix.abs().sum(dim=1, keepdim=True)
    matrix = matrix / sum_dim1
    for i in range(matrix.shape[0]):
        area = np.trapz(matrix[i], dx=1)
        matrix[i] = matrix[i] / area
    return matrix

# Find every steps that refer to the same timestamp of each cycle
def retrieve_steps_in_every_cycles(points, flat_dataset, cycles):

    # Get the cycle's length 
    dim0 = flat_dataset.shape[0]

    target_points = []
    if isinstance(cycles, int):
        cycles = [cycles]

    for cycle_length in cycles:
        #print(cycle_length)
        for point in points:
            # i=float(point % int(cycle_length))
            i=(point % int(cycle_length)).item()
            if i in target_points:
                continue 
            # #print('first point :', i)
            while int(i) < dim0:
                target_points.append(int(i))
                i += cycle_length 

    return target_points

def max_peak(vector, number_of_peaks=1, distance=1, bound=1):
    vector = vector[bound:-bound].clone()
    max_list = []
    for i in range(number_of_peaks):
        max_idx = np.argmax(vector).item()
        low = max(0,max_idx-distance)
        up = min(vector.shape[0],max_idx+distance+1)
        vector[low : up] = 0
        max_list.append(max_idx)
    max_list = np.array(max_list, dtype=int)+bound
    return max_list

def attack_contrib_elect_points(matrix, p_num):
    distance = matrix.shape[1]
    tensor_normalized = normalize_tensor(matrix)
    tensor_sum = tensor_normalized.abs().sum(dim=2)
    dim1 = tensor_sum.shape[1]
    flat_tensor_sum = tensor_sum.reshape(-1)
    
    indices = max_peak(flat_tensor_sum, number_of_peaks=p_num, distance=distance, bound=dim1**2)
    top_coords = np.array([x//dim1 + x%dim1 for x in indices])

    return top_coords

def highest_contrib_elect_points(matrix, p_num):
    dim0=matrix.shape[0]
    dim1=matrix.shape[1]
    tensor_sum_of_each_point = sum_tensor_contribution_per_point(matrix.cpu(),dim0,dim1)
    # print(f'shap_sum_of_each_point shape : {shap_sum_of_each_point.shape}')
    
    distance = dim1
    indices = max_peak(tensor_sum_of_each_point,number_of_peaks=p_num,distance=distance, bound=dim1)
    top_coords = np.array([x//dim1 + x%dim1 for x in indices])

    return top_coords

def attack_distrib_elect_points(matrix, p_num):
    distance = matrix.shape[1]
    top_coords =[]
    ## Find the highest SHAP values after norm
    tensor_sum = matrix.abs().sum(dim=2)
    dim1 = tensor_sum.shape[1]
    tensor_distrib = distrib(tensor_sum)

    flat_tensor_abs = tensor_distrib.reshape(-1)

    
    indices = max_peak(flat_tensor_abs,number_of_peaks=p_num,distance=distance, bound=dim1**2)
    top_coords = np.array([x//dim1 + x%dim1 for x in indices])
    
    return top_coords

def attack_rolling_var_elect_points(matrix, X, p_num):
    s=matrix.shape[0]
    X_flat = flatten_dataset(X).to(X.device)
        
    ## Get the main frequency component
    size = len(X_flat)
    fft_signal = np.fft.fft(X_flat)
    frequencies = np.fft.rfftfreq(size, d=1)
    idx_peak = np.argmax(np.abs(fft_signal[:size//2]))
    window = round(1/frequencies[idx_peak])

    ## Get the rolling var
    var=[]
    for t in range(s-window):
        var.append(matrix[t:t+window,:,:].cpu().var().item())
    threshold = np.percentile(var,100-p_num)
    var = torch.tensor(var)
    ## Get the target points, leveraging the highest shift in the window-local pattern
    d_zones=[]
    for t in range(s-window):
        if var[t] > threshold:
            d_zones.append(t)
    d_zones = np.array(d_zones)
    
    return d_zones

def attack_rolling_var_dynamic_elect_points(matrix, X, p_num):
    X_flat = flatten_dataset(X)
    s=matrix.shape[0]
    d_zones=[]
    roll_var = []

    size = len(X_flat)
    fft_signal = np.fft.rfft(X_flat.cpu())
    frequency_spectrum = np.abs(fft_signal)
    frequencies = np.fft.rfftfreq(size, d=1)
    indices_top_k = np.argsort(frequency_spectrum)[-p_num-2:] # To still have the number of freq needed even if you have 0 or 1
    mask  = (indices_top_k > 1)
    indices_top_k = indices_top_k[mask]
    indices_top_k = indices_top_k[:p_num]
    # print(f'indices_top_k : {indices_top_k}')
    cycles_length = [int(1/frequencies[x]) for x in indices_top_k]
    #print(f'cycles length : {cycles_length}')

    for window in cycles_length:
        ## Get the rolling var
        var=[]
        for t in range(s-window):
            var.append(matrix[t:t+window,:,:].cpu().var().item())
        point = np.argmax(var)
        var = torch.tensor(var)
        roll_var.append(var)
        d_zones.append(point)
    d_zones = np.array(d_zones)
    
    return d_zones

############ Attacks ############

class Attack:

    def __init__(self, cln_input, ground_truth, cln_predictions, tensor, model, dev, method_name, p_num, eps):
        # self.cln_input = cln_input.clone().detach().requires_grad_(True)
        # self.ground_truth = ground_truth
        # self.model = model
        # cln_input.device = device
        # self.config = config

        method = getattr(self, method_name, None)
        if method_name in ['pgd', 'fgsm', 'bim']:
            X_adv, output, target_points, decision_tensor, indices, X_indices, duration = method(cln_input, ground_truth, model, dev, p_num, eps)
        else:
            X_adv, output, target_points, decision_tensor, indices, X_indices, duration = method(cln_input, tensor, model, dev, p_num, eps)

        self.input = X_adv.detach()
        self.output = output
        self.target_points = target_points
        self.decision_tensor = decision_tensor
        self.X_indices = X_indices
        self.indices = indices
        self.duration = duration

        self.in_fnorm = torch.norm(cln_input - self.input, p='fro').item()
        self.in_mse = F.mse_loss(cln_input.squeeze(),self.input.squeeze()).item()
        self.in_mape = (torch.mean(torch.abs((cln_input.squeeze() - self.input.squeeze()) / (cln_input.squeeze()))) * 100).item()
        self.out_mse_gt = F.mse_loss(ground_truth.squeeze(), self.output.squeeze()).item()
        self.out_mae_gt = F.l1_loss(ground_truth.squeeze(), self.output.squeeze()).item()
        self.out_mse = F.mse_loss(cln_predictions.squeeze(), self.output.squeeze()).item()
        self.out_mae = F.l1_loss(cln_predictions.squeeze(), self.output.squeeze()).item()

        self.num_targeted = len(self.target_points)

    def get_attack_info(self):
        return self.input, self.output, self.target_points, self.decision_tensor

    def get_metrics(self):
        return [self.in_fnorm, self.in_mse, self.in_mape, self.out_mse, self.out_mae, self.num_targeted]

    def fgsm(self, cln_input, ground_truth, model, dev, num_iter, epsilon):
        model.eval()
        start = time.time()
        X_adv = cln_input.clone().to(dev) # Track all operations on a tensor to compute gradients during backpropagation
        X_adv.requires_grad_(True)
        output = model(X_adv).cpu()
        loss = F.mse_loss(output, ground_truth)
        loss.backward() # backward allows the .grad field of any tensor that has .requires_grad_ set to True to be filled with the gradient

        perturbation = epsilon * X_adv.grad.data.sign() # Returns a new tensor containing the sign of the gradient but detached from the computation graph
        X_adv = X_adv + perturbation
        
        end = time.time()

        X_adv= X_adv.cpu()
        difference = flatten_dataset(X_adv) != flatten_dataset(cln_input)
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
            
        return X_adv.detach(), output, target_points, perturbation, np.array([]), np.array([]), end - start 


    # Usually alpha is set between eps/num_iter and epsilon
    def bim(self, cln_input, ground_truth, model, dev, num_iter, epsilon):
        model.eval()
        start = time.time()
        X_adv = cln_input.clone().to(dev)
        alpha = 0.01
        perturbation = []

        for _ in range(num_iter):
            X_adv.requires_grad_(True)
            output = model(X_adv).cpu()

            model.zero_grad() ## Prevent Gradient accumulation
            loss = F.mse_loss(output, ground_truth)
            loss.backward()

            grad_sign = X_adv.grad.data.sign()
            perturbation = alpha * grad_sign

            X_adv = (X_adv + perturbation).detach()
        
        end = time.time()
        X_adv= X_adv.cpu()
        difference = flatten_dataset(X_adv) != flatten_dataset(cln_input)
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
            
        return X_adv.detach(), output, target_points, perturbation, np.array([]), np.array([]), end - start  
    
    def pgd(self, cln_input, ground_truth, model, dev, iters, epsilon):
        model.eval()
        start = time.time()
        alpha = 1
        X_adv = cln_input.clone()
        rand_start = torch.empty_like(cln_input).uniform_(-epsilon, epsilon)
        X_adv = X_adv + rand_start
        perturbation = []
        X_adv = X_adv.to(dev)
        for _ in range(iters):
            X_adv.requires_grad = True
            output = model(X_adv).cpu()
            model.zero_grad()
            loss = F.mse_loss(output, ground_truth)
            loss.backward()

            # Gradient step
            grad_sign = X_adv.grad.data.sign()
            perturbation = alpha * grad_sign

            # Project back to the epsilon-ball
            X_adv = torch.clamp(X_adv + perturbation, min=X_adv-epsilon, max=X_adv+epsilon).detach()
        
        end = time.time()

        X_adv= X_adv.cpu()
        difference = flatten_dataset(X_adv) != flatten_dataset(cln_input)
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]

        return X_adv, output, target_points, perturbation, np.array([]), np.array([]), end - start 
    
    def AutoPGD(self, cln_input, ground_truth, model, dev, iters, epsilon):
        model.eval()
        start = time.time()
        X_adv = cln_input.clone().to(dev)

        attack = APGDAttack(model, iter=10)
        X_adv = attack.perturb(X_adv,ground_truth,True)

        end = time.time()
        difference = flatten_dataset(X_adv) != flatten_dataset(cln_input)
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, np.array([]), np.array([]), np.array([]), end - start 
    
    def mAutoPGD_TSF(self, cln_input, ground_truth, model, dev, iters, epsilon):
        # (
        # model,                  # regression model (e.g. LSTM/Transformer)
        # x0,                     # original input (batch_size, seq_len, features)
        # target_y,              # target prediction (same shape as model(x0))
        # loss_fn=F.mse_loss,     # loss function (default: MSE)
        # eps=0.1,                # max L_infinity perturbation (optional)
        # alpha_init=0.01,        # initial step size
        # beta=0.75,              # momentum parameter
        # n_iter=40,              # number of iterations
        # check_points=[10, 20, 30],  # checkpoints for adaptive step size
        # condition_fn=None,      # custom condition for adapting step size
        # clamp_min=0.0,
        # clamp_max=1.0,
        # project=True            # whether to project into eps-ball
        # ):
        project = True # whether to project into eps-ball
        beta = 0.75 # momentum parameter
        check_points=[10, 20, 30],  # checkpoints for adaptive step size
        condition_fn = None
        clamp_min, clamp_max = 0,1
        start = time.time()
        x = cln_input.clone().detach().requires_grad_(True)
        x_prev = x.clone().detach()
        alpha = 0.01

        # Initial forward and loss
        output = model(x)
        loss = F.mse_loss(output, ground_truth)
        grad = torch.autograd.grad(loss, x)[0]
        z = x - alpha * grad
        if project:
            z = torch.clamp(z, x - epsilon, x + epsilon)
        z = torch.clamp(z, clamp_min, clamp_max)

        # Determine initial xmin and fmin
        output_z = model(z)
        loss_z = F.mse_loss(output_z, ground_truth)
        if loss_z < loss:
            xmin, fmin = z.detach(), output_z.detach()
        else:
            xmin, fmin = x.detach(), output.detach()

        for n in range(1, iters):
            x = x.detach().requires_grad_(True)
            output = model(x)
            loss = F.mse_loss(output, ground_truth)
            grad = torch.autograd.grad(loss, x)[0]

            # First update (z)
            z = x - alpha * grad
            if project:
                z = torch.clamp(z, x - epsilon, x + epsilon)
            z = torch.clamp(z, clamp_min, clamp_max)

            # Second update with momentum
            X_adv = x + beta * (z - x) + (1 - beta) * (x - x_prev)
            if project:
                X_adv = torch.clamp(X_adv, x - epsilon, x + epsilon)
            X_adv = torch.clamp(X_adv, clamp_min, clamp_max)

            # Evaluate new candidate
            with torch.no_grad():
                output_new = model(X_adv)
                loss_new = F.mse_loss(output_new, ground_truth)

                if loss_new < F.mse_loss(fmin, ground_truth):
                    xmin = X_adv.clone().detach()
                    fmin = output_new.clone().detach()

            # Adaptive step size
            if n in check_points:
                if condition_fn is not None and condition_fn(loss_new, output_new, ground_truth):
                    alpha /= 2.0
                    X_adv = xmin.clone().detach()

            x_prev = x.clone().detach()
            x = X_adv.clone().detach().requires_grad_(True)

        end = time.time()
        difference = flatten_dataset(X_adv) != flatten_dataset(cln_input)
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, np.array([]), np.array([]), np.array([]), end - start 

    # @staticmethod
    def attack_contrib(self, cln_input, tensor, model, dev, p_num, eps):
        """
        cln_input : [N,L]
        Tensor : [N,L,H]
        It compares every points (in the N*L points) that correspond to the same timestamp (number of timestamps N + L) using the sum of their 
        H contributions. 
        For a fair comparison between the different timestamp, it devides each points by the sum of every H contributions that correspond to 
        the same timestamp.
        Then the points elected are those which have the greatest difference between them and the other points corresponding to the same timestamp.
        """
        start = time.time()
        X_flat = flatten_dataset(cln_input)
        matrix = tensor.clone().detach()

        ## Get the main frequency component
        size = len(X_flat)
        fft_signal = np.fft.fft(X_flat)
        frequencies = np.fft.rfftfreq(size, d=1)
        idx_peak = np.argmax(np.abs(fft_signal[:size//2]))
        cycle_length = round(1/frequencies[idx_peak])

        ## Find the highest SHAP values after norm
        distance = matrix.shape[1]
        tensor_normalized = normalize_tensor(matrix)
        tensor_sum = tensor_normalized.abs().sum(dim=2)
        dim1 = tensor_sum.shape[1]
        flat_tensor_sum = tensor_sum.reshape(-1)
        
        indices = max_peak(flat_tensor_sum, number_of_peaks=p_num, distance=distance, bound=dim1**2)
        # print("indices : ", indices)
        # max_values, indices_values = torch.topk(flat_tensor_sum, p_num)
        # print("indices_values : ", indices_values)
        # top_coords = torch.stack(torch.unravel_index(indices_values, flat_tensor_sum.shape), dim=1)[:,0].cpu().numpy()
        # print("top_coords : ", top_coords)
        top_coords = np.array([x//dim1 + x%dim1 for x in indices])
        
        ## Get the target points, leveraging the top coordinates and the seasonality of the dataset
        target_points = retrieve_steps_in_every_cycles(top_coords, X_flat, cycle_length)

        ## Poison the dataset
        X_adv = inject_traffic(X_flat, target_points, 1, eps)
        difference = X_adv != X_flat
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        X_adv = reshape_dataset(X_adv, cln_input)
        X_adv = X_adv.to(dev)
        end = time.time()
        X_indices = inject_traffic(X_flat, top_coords, 1, eps)
        X_indices = reshape_dataset(X_indices, cln_input)
        model.eval()
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, flat_tensor_sum, top_coords, X_indices, end - start 

    # @staticmethod
    def highest_contrib(self, cln_input, tensor, model, dev, p_num, eps):
        """
        cln_input : [N,L]
        Tensor : [N,L,H]
        It sums every points (in the N*L points) that correspond to the same timestamp (number of timestamps N + L) using the sum of their 
        H contributions.
        Then the points elected are the timestamps that have the highest loopback contribution.
        """
        start = time.time()
        X_flat = flatten_dataset(cln_input)
        matrix = tensor.clone().detach()

        ## Get the main frequency component
        size = len(X_flat)
        fft_signal = np.fft.fft(X_flat)
        frequencies = np.fft.rfftfreq(size, d=1)
        idx_peak = np.argmax(np.abs(fft_signal[:size//2]))
        cycle_length = round(1/frequencies[idx_peak])

        dim0=matrix.shape[0]
        dim1=matrix.shape[1]
        tensor_sum_of_each_point = sum_tensor_contribution_per_point(matrix,dim0,dim1)
        # print(f'shap_sum_of_each_point shape : {shap_sum_of_each_point.shape}')
        
        distance = dim1
        indices = max_peak(tensor_sum_of_each_point,number_of_peaks=p_num,distance=distance, bound=dim1)
        top_coords = np.array([x//dim1 + x%dim1 for x in indices])
        # max_values, indices_values = torch.topk(tensor_sum_of_each_point, p_num)
        ## Get the target points, leveraging the top coordinates and the seasonality of the dataset
        target_points = retrieve_steps_in_every_cycles(top_coords, X_flat, cycle_length)

        ## Poison the dataset
        X_adv = inject_traffic(X_flat, target_points, 1, eps)
        difference = X_adv != X_flat
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        X_adv = reshape_dataset(X_adv, cln_input)
        X_adv = X_adv.to(dev)
        end  = time.time()
        X_indices = inject_traffic(X_flat, top_coords, 1, eps)
        X_indices = reshape_dataset(X_indices, cln_input)
        model.eval()
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, tensor_sum_of_each_point, top_coords, X_indices, end - start


    # @staticmethod
    def attack_distrib(self, cln_input, tensor, model, dev, p_num, eps):
        """
        cln_input : [N,L]
        Tensor : [N,L,H]
        It compares every timestamps in a single slice (N) using the sum of their H contributions. 
        For a fair comparison between the different slice, it devides each points by the sum of every H contributions that correspond to 
        the same slice.
        Then the points elected are those which have the greatest difference between them and the other points corresponding to the same slice.
        """
        start  = time.time()
        X_flat = flatten_dataset(cln_input)

        ## Get the main frequency component
        size = len(X_flat)
        fft_signal = np.fft.fft(X_flat)
        frequencies = np.fft.rfftfreq(size, d=1)
        idx_peak = np.argmax(np.abs(fft_signal[:size//2]))
        cycle_length = round(1/frequencies[idx_peak])

        matrix = tensor.clone().detach()
        distance = matrix.shape[1]
        top_coords =[]
        ## Find the highest SHAP values after norm
        tensor_sum = matrix.abs().sum(dim=2)
        dim1 = tensor_sum.shape[1]
        tensor_distrib = distrib(tensor_sum)

        flat_tensor_abs = tensor_distrib.reshape(-1)
 
        
        indices = max_peak(flat_tensor_abs,number_of_peaks=p_num,distance=distance, bound=dim1**2)
        top_coords = np.array([x//dim1 + x%dim1 for x in indices])
        # max_values_abs, indices_values = torch.topk(flat_tensor_abs, p_num)
        # top_coords = torch.stack(torch.unravel_index(indices_values, tensor_distrib.shape), dim=1)[:,0].cpu().numpy()

        ## Get the target points, leveraging the top coordinates and the seasonality of the dataset
        target_points = retrieve_steps_in_every_cycles(top_coords, X_flat, cycle_length)
    
        ## Poison the dataset
        X_adv = inject_traffic(X_flat, target_points, 1, eps)
        difference = X_adv != X_flat
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        X_adv = reshape_dataset(X_adv, cln_input)
        X_adv = X_adv.to(dev)
        end  = time.time()
        X_indices = inject_traffic(X_flat, top_coords, 1, eps)
        X_indices = reshape_dataset(X_indices, cln_input)
        model.eval()
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, flat_tensor_abs, top_coords, X_indices, end - start


    # @staticmethod
    def attack_rolling_var(self, cln_input, tensor, model, dev, p_num, eps):
        """
        cln_input : [N,L]
        Tensor : [N,L,H]
        It computes the rolling variance of input dataset using a window equals to the 
        main frequency component.
        The points elected are those which have the greatest rolling variance.
        """
        start  = time.time()
        X_flat = flatten_dataset(cln_input)
        
        ## Get the main frequency component
        size = len(X_flat)
        fft_signal = np.fft.fft(X_flat)
        frequencies = np.fft.rfftfreq(size, d=1)
        idx_peak = np.argmax(np.abs(fft_signal[:size//2]))
        window = round(1/frequencies[idx_peak])

        s=cln_input.shape[0]
        matrix = tensor.clone().detach()

        ## Get the rolling var
        var=[]
        for t in range(s-window):
            var.append(matrix[t:t+window,:,:].var().item())
        threshold = np.percentile(var,100-p_num)
        var = torch.tensor(var)
        ## Get the target points, leveraging the highest shift in the window-local pattern
        d_zones=[]
        for t in range(s-window):
            if var[t] > threshold:
                d_zones.append(t)
        d_zones = np.array(d_zones)
        ## Get the target points, leveraging the top coordinates and the seasonality of the dataset
        target_points = retrieve_steps_in_every_cycles(d_zones, X_flat, window)

        ## Poison the dataset
        X_adv = inject_traffic(X_flat, target_points, 1, eps)
        difference = X_adv != X_flat
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        X_adv = reshape_dataset(X_adv, cln_input)
        X_adv = X_adv.to(dev)
        end  = time.time()
        X_indices = inject_traffic(X_flat, d_zones, 1, eps)
        X_indices = reshape_dataset(X_indices, cln_input)
        model.eval()
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        return X_adv.detach(), output, target_points, var, d_zones, X_indices, end - start
    
    # @staticmethod
    def attack_rolling_var_dynamic(self, cln_input, tensor, model, dev, p_num, eps):
        """
        cln_input : [N,L]
        Tensor : [N,L,H]
        It computes the rolling variance of input dataset using windows equal to the 
        most important frequency components.
        The points elected are those which have the greatest rolling variance. This
        is repeated for each frequency component.
        """
        start  = time.time()
        X_flat = flatten_dataset(cln_input)

        s=tensor.shape[0]
        d_zones=[]
        roll_var = []
        matrix = tensor.clone().detach()

        size = len(X_flat)
        fft_signal = np.fft.rfft(X_flat)
        frequency_spectrum = np.abs(fft_signal)
        frequencies = np.fft.rfftfreq(size, d=1)
        indices_top_k = np.argsort(frequency_spectrum)[-p_num-2:] # To still have the number of freq needed even if you have 0 or 1
        mask  = (indices_top_k > 1)
        indices_top_k = indices_top_k[mask]
        indices_top_k = indices_top_k[:p_num]
        # print(f'indices_top_k : {indices_top_k}')
        cycles_length = [int(1/frequencies[x]) for x in indices_top_k]
        #print(f'cycles length : {cycles_length}')

        for window in cycles_length:
            ## Get the rolling var
            var=[]
            for t in range(s-window):
                var.append(matrix[t:t+window,:,:].var().item())
            point = np.argmax(var)
            var = torch.tensor(var)
            roll_var.append(var)
            d_zones.append(point)
        d_zones = np.array(d_zones)

        ## Get the target points, leveraging the top coordinates and the seasonality of the dataset
        target_points = retrieve_steps_in_every_cycles(d_zones, X_flat, cycles_length)

        ## Poison the dataset
        X_adv = inject_traffic(X_flat, target_points, 1, eps)
        difference = X_adv != X_flat
        diff_indices = torch.nonzero(difference, as_tuple=False)
        target_points = [idx.item() for idx in diff_indices]
        X_adv = reshape_dataset(X_adv, cln_input)
        X_adv = X_adv.to(dev)
        end  = time.time()
        X_indices = inject_traffic(X_flat, d_zones, 1, eps)
        X_indices = reshape_dataset(X_indices, cln_input)
        model.eval()
        output = model(X_adv).cpu()
        X_adv= X_adv.cpu()

        # print(X_adv.shape, output.shape, len(target_points), len(roll_var))
        return X_adv.detach(), output, target_points, roll_var, d_zones, X_indices, end - start
    