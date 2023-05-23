import torch
import torch.nn as nn
import numpy as np
import math
import scipy.optimize as sopt
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 
import SUGAR.utils as ut
torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3)


# =================================================================================================================
# This contains the helper functions for the implementation of the structure learning of DAG.
# The credits of the codes should be given to: https://github.com/xunzheng/notears
# =================================================================================================================



class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        """
        This initialize the class.

        Input
        ----------
        num_linear: num of local linear layers
        input_features: the input features into the model
        output_features: the output features from the model

        Output
        ----------
        None
        """
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        
        # initialize the weights and bias of the neural network
        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features).to(device))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
    
class LBFGSBScipy(torch.optim.Optimizer):
    """Wrap L-BFGS-B algorithm, using scipy routines.
    
    Courtesy: Arthur Mensch's gist
    https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
    """

    def __init__(self, params):
        defaults = dict()
        super(LBFGSBScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSBScipy doesn't support per-parameter options"
                             " (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel = sum([p.numel() for p in self._params])

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_().to(device)
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1).to(device)
            else:
                view = p.grad.data.view(-1).to(device)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_bounds(self):
        bounds = []
        for p in self._params:
            if hasattr(p, 'bounds'):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1).to(device)
            else:
                view = p.data.view(-1).to(device)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params).to(device)
            flat_params = flat_params.to(torch.get_default_dtype())
            self._distribute_flat_params(flat_params)
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype('float64')

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()

        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds)

        final_params = torch.from_numpy(sol.x).to(device)
        final_params = final_params.to(torch.get_default_dtype())
        self._distribute_flat_params(final_params)

        
class NotearsMLP(nn.Module):
    """
    This class is to define the module of the Notears MLP method for DAG structual learning
    """
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x).to(device)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(d).to(device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1).to(device)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2).to(device)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2).to(device)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight).to(device)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t().to(device)  # [i, j]
        W = torch.sqrt(A).to(device)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
    
    
def squared_loss(output, target):
    # calculate the squared loss
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """
    This function Perform one step of dual ascent in augmented Lagrangian..
    
    Input
    ----------
    model: the neural network model
    X: input sample matrix
    lambda1, lambda2: hyperparameters for regulizations
    rho, alpha, h: convergence variables for optimization
    rho_max: upper bound to avoid numerical issues
    
    Output
    ----------
    rho, alpha, h_new: return the convergence variables for optimization to get the information of stopping
    """
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X).to(device)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    """
    This function implements the Notears MLP method for DAG structual learning.
    
    Input
    ----------
    X: input sample matrix
    lambda1, lambda2: hyperparameters for regulizations
    max_iter: maximum number of iterations
    h_tol: threshold value of convergence
    rho_max: upper bound to avoid numerical issues
    w_threshold: graph weights cutoff
    
    Output
    ----------
    W_est: the estimated DAG graph
    """
    
    # specify the convergence variables
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        print(_)
        # update the model in each iteration
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    # convert the result to graph edge weights
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est