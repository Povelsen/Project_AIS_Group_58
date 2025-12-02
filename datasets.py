"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        seq[:seqlen,:] = m_v[:seqlen,:]
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        
        return seq , mask, seqlen, mmsi, time_start
    
class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """
    def __init__(self, 
                 l_data, 
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        # lat and lon
        seq[:seqlen,:2] = m_v[:seqlen,:2] 
        # dlat and dlon
        dpos = (m_v[1:,:2]-m_v[:-1,:2]+self.dpos_max )/(2*self.dpos_max )
        dpos = np.concatenate((dpos[:1,:],dpos),axis=0)
        dpos[dpos>=1] = 0.9999
        dpos[dpos<=0] = 0.0
        seq[:seqlen,2:] = dpos[:seqlen,:2] 
        
        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        
        return seq , mask, seqlen, mmsi, time_start

class AISDataset_Enhanced(Dataset):
    """Dataset that includes port and coastline distance features + port labels."""
    
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 use_port_features=True,
                 use_coast_features=True,
                 predict_ports=False,
                 data_augmentation=False,
                 aug_noise_std=0.005,
                 aug_temporal_jitter=True,
                 device=torch.device("cpu")):
        """
        Args:
            l_data: list of dictionaries with:
                l_data[idx]["traj"]: matrix with columns:
                    [LAT, LON, SOG, COG, TIMESTAMP, MMSI, PORT_FEAT, COAST_FEAT, PORT_LABEL]
            predict_ports: whether to return port labels for training
        """
        self.max_seqlen = max_seqlen
        self.device = device
        self.l_data = l_data
        self.use_port = use_port_features
        self.use_coast = use_coast_features
        self.predict_ports = predict_ports
        self.data_augmentation = data_augmentation
        self.aug_noise_std = aug_noise_std
        self.aug_temporal_jitter = aug_temporal_jitter
        
        # Calculate feature dimension
        self.base_dim = 4  # lat, lon, sog, cog
        self.feature_dim = self.base_dim
        if self.use_port:
            self.feature_dim += 1
        if self.use_coast:
            self.feature_dim += 1

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Returns sequence with optional port/coast features and port labels."""
        V = self.l_data[idx]
        
        # Base features
        m_base = V["traj"][:, :4]
        m_base[m_base > 0.9999] = 0.9999
        
        seqlen = min(len(m_base), self.max_seqlen)
        
        # Prepare arrays
        seq = np.zeros((self.max_seqlen, self.feature_dim))
        
        # --- STEP 1: LOAD RAW LABELS EARLY (Before Augmentation) ---
        raw_labels = None
        if self.predict_ports and V["traj"].shape[1] > 8:
            # Init with -100 (ignore index)
            raw_labels = np.full(self.max_seqlen, -100, dtype=np.int64)
            # Load actual data
            raw_labels[:seqlen] = V["traj"][:seqlen, 8].astype(np.int64)
            
        # Add base features
        seq[:seqlen, :4] = m_base[:seqlen, :]
        
        # --- STEP 2: APPLY AUGMENTATION ---
        if self.data_augmentation:
            # A. Gaussian Noise
            noise = np.random.normal(0, self.aug_noise_std, size=(seqlen, 2))
            seq[:seqlen, :2] += noise
            seq[:seqlen, :2] = np.clip(seq[:seqlen, :2], 0.0, 0.9999)
            
            # B. Temporal Jitter (Shift)
            if self.aug_temporal_jitter and seqlen > 10:
                max_skip = min(5, seqlen - 10)
                skip = np.random.randint(0, max_skip + 1)
                if skip > 0:
                    # Shift Features
                    seq[:-skip, :] = seq[skip:, :]
                    seq[-skip:, :] = 0
                    
                    # Shift Labels (CRITICAL FIX: Keep labels synced with features)
                    if raw_labels is not None:
                        raw_labels[:-skip] = raw_labels[skip:]
                        raw_labels[-skip:] = -100 # invalid after shift
                    
                    seqlen -= skip

        # Add optional features
        col_idx = 4
        if self.use_port:
            port_feat = V["traj"][:, 6]
            seq[:seqlen, col_idx] = port_feat[:seqlen]
            col_idx += 1
            
        if self.use_coast:
            coast_feat = V["traj"][:, 7]
            seq[:seqlen, col_idx] = coast_feat[:seqlen]
        
        # Convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen_tensor = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        
        # --- STEP 3: RETURN LABELS WITH SAFETY CLIPPING ---
        if raw_labels is not None:
            # Safety: Clip labels to valid range [0, 511]
            # Max ID in data is 508. Config port_size is 512.
            # Valid indices are 0 to 511.
            
            MAX_VALID_PORT = 511  # <--- UPDATED HERE
            
            # Logic: If it is NOT -100, clip it to 511.
            valid_mask = (raw_labels != -100)
            raw_labels[valid_mask] = np.clip(raw_labels[valid_mask], 0, MAX_VALID_PORT)
            
            port_labels = torch.tensor(raw_labels, dtype=torch.long)
            
            return seq, mask, seqlen_tensor, mmsi, time_start, port_labels
        else:
            return seq, mask, seqlen_tensor, mmsi, time_start