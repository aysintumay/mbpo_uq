import numpy as np
import torch
from common import util


class StandardNormalizer(object):
    def __init__(self):
        self.tot_count = 0
    
    def reset(self):
        self.mean = None
        self.var = None
        self.tot_count = 0

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        if isinstance(data, torch.Tensor):
            self.mean = torch.mean(data, dim=0, keepdims=True).to(util.device)
            self.var = torch.var(data, dim=0, keepdims=True).to(util.device)
        elif isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.var = np.var(data, axis=0, keepdims=True)
        self.var[self.var < 1e-12] = 1.0
        self.tot_count = len(data)

    def update(self, samples):
        sample_count = len(samples)
        #initialize
        if self.tot_count == 0:
            dim = samples.shape[1]
            if isinstance(samples, torch.Tensor):
                self.mean = torch.zeros(dim, dtype=torch.float32).to(util.device)
                self.var = torch.ones(dim, dtype=torch.float32).to(util.device)
            elif isinstance(samples, np.ndarray):
                self.mean = np.zeros(dim, dtype=float)
                self.var = np.ones(dim, dtype=float)

        if isinstance(samples, torch.Tensor):
            sample_mean = torch.mean(samples, dim=0, keepdims=True)
            sample_var = torch.var(samples, dim=0, keepdims=True)
        elif isinstance(samples, np.ndarray):
            sample_mean = np.mean(samples, axis=0, keepdims=True)
            sample_var = np.var(samples, axis=0, keepdims=True)

        delta_mean = sample_mean - self.mean

        new_mean = self.mean + delta_mean * sample_count / (self.tot_count + sample_count)
        prev_var = self.var * self.tot_count
        sample_var = sample_var * sample_count
        new_var = prev_var + sample_var + delta_mean * delta_mean * self.tot_count * sample_count / (self.tot_count + sample_count)
        new_var = new_var / (self.tot_count + sample_count)

        self.mean = new_mean
        self.var = new_var
        self.var[self.var < 1e-12] = 1.0
        self.tot_count += sample_count

    def transform(self, data):
        return data
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if self.mean is None or self.var is None:
            return data
        if isinstance(data, torch.Tensor):
            return (data - torch.tensor(self.mean).to(data.device)) / torch.sqrt(torch.tensor(self.var).to(data.device))
        elif isinstance(data, np.ndarray):
            return (data - np.array(self.mean)) / np.sqrt(np.array(self.var))

    def inverse_transform(self, data):
        return data
        if self.mean is None or self.var is None:
            return data
        if isinstance(data, torch.Tensor):
            return data * torch.sqrt(torch.tensor(self.var).to(data.device)) + torch.tensor(self.mean).to(data.device)
        elif isinstance(data, np.ndarray):
            return data * np.sqrt(np.array(self.var)) + np.array(self.mean)

