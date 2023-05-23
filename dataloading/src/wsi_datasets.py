import os
import pandas as pd
from pathlib import Path
from openslide import OpenSlide
import numpy as np
from torch.utils.data import Dataset
import torch

class WSI_unsupervised(Dataset):
    def __init__(self,wsi_folder=Path('WSI_download/WSI_data'),tile_size=(1024,1024),offset=768,transform=None,
                    wsi_file_exts=['.tiff','.svs']):
        
        self.tile_size,_=tile_size
        self.offset=offset
        self.overlap=self.tile_size-self.offset
        assert(self.overlap>=0),'offset must be less than tile_size'
        self.wsi_fpaths=[wsi_folder/file for file in os.listdir(wsi_folder) if (wsi_folder/file).suffix in file_exts]
        self.wsi_props=pd.DataFrame({'filepath':self.wsi_fpaths,'dims':[OpenSlide(fpath).dimensions for fpath in self.wsi_fpaths]})
        
        ## find number of tiles per row as we slide  a window of tile_size,tile_size across the WSI
        self.wsi_props['num_tiles_row']=[(W-tile_size)//offset  for W,_ in self.wsi_props['dims']]
        self.wsi_props['num_tiles_col']=[(H-tile_size)//offset  for _,H in self.wsi_props['dims']]
        self.wsi_props['num_tiles']=self.wsi_props['num_tiles_row']* self.wsi_props['num_tiles_col']
        self.wsi_props['tile_end_idx']=self.wsi_props['num_tiles'].cumsum()
        ## get the indices of the entire dataset across all images
        self.dataset_idxs=list(range(self.wsi_props['tile_end_idx'].iloc[-1]))
        
        self.transform = transform
        
                                         
        ## define unique indexes for each tile of size * size in the entire dataset
                                      
        

    def __len__(self):
        return len(self.dataset_idxs)

    def __getitem__(self, idx):
        tile_idx=self.dataset_idxs[idx]
        
         ## find the .tiff image the index tile belongs to
        wsi_idx=self.wsi_props['tile_end_idx'].searchsorted(tile_idx)
        wsi_fpath=self.wsi_props['filepath'].iloc[wsi_idx]
        ## get number of tiles till previous whole slide image 
        prev_end=self.wsi_props['tile_end_idx'].iloc[wsi_idx-1]
        # get the index of the selected tile in the WSI it belongs to
        # (convert dataset index to WSI index)
        tile_idx=tile_idx-prev_end
        num_tiles_row=self.wsi_props['num_tiles_row'].iloc[wsi_idx]
        tile_offset_X,tile_offset_Y=tile_idx%num_tiles_row,tile_idx//num_tiles_row
        # finally get tile location (top left) in WSI coordinates to pass to openslide for extraction
        tile_top_left=tile_offset_X*self.offset,tile_offset_Y*self.offset
        # create slide object and extract tile
        slide=OpenSlide(wsi_fpath)
        tile_arr=np.array(slide.read_region(tile_top_left,0,self.tile_size))
        # get tile rgb array,discard the alpha values -which are constant anyway
        tile_rgb=tile_arr[:,:,:-1]
        tile_T=torch.tensor(tile_rgb)
        # reshape to C X H  X W for CNNs
        return torch.permute(tile_T,(2,1,0))
        
        
        
        
        
        
        
         
        
        
       
