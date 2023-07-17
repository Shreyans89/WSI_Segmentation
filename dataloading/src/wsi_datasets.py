import os
import pandas as pd
from pathlib import Path
from openslide import OpenSlide
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision.utils import make_grid
from functools import partial
from shapely.geometry import Polygon,MultiPolygon
import  cv2
import openslide


class WSI_Unsupervised(Dataset):
    def __init__(self,wsi_folder=Path('WSI_download/WSI_data'),tile_size=(1024,1024),offset=768,transform=None,
                    wsi_file_exts=['.tiff','.svs']):
        
        self.tile_size,_=tile_size
        self.offset=offset
        self.overlap=self.tile_size-self.offset
        assert(self.overlap>=0),'offset must be less than tile_size'
        self.wsi_fpaths=[wsi_folder/file for file in os.listdir(wsi_folder) if (wsi_folder/file).suffix in wsi_file_exts]
        self.wsi_props=pd.DataFrame({'filepath':self.wsi_fpaths,'dims':[OpenSlide(fpath).dimensions for fpath in                   self.wsi_fpaths]})
        
        ## find number of tiles per row as we slide  a window of tile_size,tile_size across the WSI
        self.wsi_props['num_tiles_row']=[(W-self.tile_size)//offset  for W,_ in self.wsi_props['dims']]
        self.wsi_props['num_tiles_col']=[(H-self.tile_size)//offset  for _,H in self.wsi_props['dims']]
        self.wsi_props['num_tiles']=self.wsi_props['num_tiles_row']* self.wsi_props['num_tiles_col']
        self.wsi_props['tile_end_idx']=self.wsi_props['num_tiles'].cumsum()
        self.wsi_props['prev_tile_end']=pd.concat([pd.Series(0),self.wsi_props['tile_end_idx'].iloc[:-1]]).values
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
        prev_end=self.wsi_props['prev_tile_end'].iloc[wsi_idx]
        # get the index of the selected tile in the WSI it belongs to
        # (convert dataset index to WSI index)
        tile_idx=tile_idx-prev_end
        num_tiles_row=self.wsi_props['num_tiles_row'].iloc[wsi_idx]
        tile_offset_X,tile_offset_Y=tile_idx%num_tiles_row,tile_idx//num_tiles_row
        # finally get tile location (top left) in WSI coordinates to pass to openslide for extraction
        tile_top_left=(tile_offset_X*self.offset,tile_offset_Y*self.offset)
        # create slide object and extract tile
        slide=OpenSlide(wsi_fpath)
        #pdb.set_trace()
        tile_arr=np.array(slide.read_region(tile_top_left,0,(self.tile_size,self.tile_size)))
        # get tile rgb array,discard the alpha values -which are constant anyway
        tile_rgb=tile_arr[:,:,:-1]
        tile_T=torch.tensor(tile_rgb)
        # reshape to C X H  X W for CNNs
        return torch.permute(tile_T,(2,1,0))
        
        
        
        
class WSI_Multiscale(Dataset):
    
   
    
    def __init__(self,
                 anno_df:pd.DataFrame,
                 crop_pixel_size:tuple=(256,256),
                 transform=None)->None:
       
        """Pytorch Dataset class representing a multiscale WSI dataset.inputs are img and anno dfs containing info 
         about WSI images and annotations"""
       
        self.anno_df=anno_df
        ## the size in pixel of  each crop -size is kept same at 
        ## various zoom levels for batching
        self.crop_pixel_size=crop_pixel_size
        self.item_transform = transform
        self.class2num={'Tumor':1}
        

    def __len__(self):
        return len(self.anno_df)
    
    
    def get_random_crop(self,annotation_row:pd.Series)->(torch.tensor,torch.tensor):
        
        """ get a random center crop at any possible zoom level from the periphery
           of an annotation """
       
        downsample_dict=annotation_row['downsample_levels']
        wsi_size,anno_coordinates=(torch.tensor(x) for x in [annotation_row['WSI_size'],
                                                             annotation_row['coordinates']])

        downsample_factors=torch.tensor(list(downsample_dict.values()))


        ## create on self.device
        offsets=torch.tensor(self.crop_pixel_size)//2
        
        offsets_arr=torch.tensor([[-1,1],[1,1],[1,-1],[-1,-1]])*offsets

        ## create offsets to be made from each polygon boundary point

        all_offsets=offsets_arr.unsqueeze(2)*downsample_factors.unsqueeze(0).unsqueeze(0)

        all_crops=all_offsets.unsqueeze(0)+anno_coordinates.unsqueeze(1).unsqueeze(3)
        all_crops=all_crops.permute(0,3,1,2)
        all_crops=all_crops.flatten(end_dim=1)
        

        ## all_crops shape: N_points X Zoom factors X 4 (square tile points) X 2 (x,y coords)

        ## check the boundary of the crops fall within the WSI image
        max_bounds=torch.max(all_crops,axis=1).values<wsi_size.unsqueeze(0)
        min_bounds=torch.min(all_crops,axis=1).values>torch.zeros_like(wsi_size.unsqueeze(0))

        ## get all feasible crops/tiles associated with theat particular annotation
        all_crops=all_crops[torch.logical_and(max_bounds.all(axis=1),min_bounds.all(axis=1))]

       
        random_crop_index=np.random.randint(0,all_crops.shape[0])
        
        ## select the random crop from one of the  vertices and zoom level
        crop=all_crops[random_crop_index]
        crop_sz=crop[:,0].max()-crop[:,0].min()
       
        ## record the zoom level of the chosen crop
        zoom_level_idx=torch.argwhere(downsample_factors*self.crop_pixel_size[0]==crop_sz).flatten()
       
       
        ## return the crop as a tensor of shape 4X2 (with corner coordinates of the crop) and 
        ## zoom level indicated by an index (0-8)
        return np.array(crop),zoom_level_idx
    
    
    def get_intersecting_polys(self,crop_poly:Polygon,anno_poly:Polygon,top_left:np.ndarray):
        """return the intersection between a crop square and annotation polygon
         handles the case when the intersection is a multipolygon"""
        
        intersect=crop_poly.intersection(poly)
        
        
        
        
   
    def get_mask_per_class(self,class_annotation_data:pd.DataFrame,crop:np.ndarray,
                          downsample_factor:float)->torch.tensor:
       
        """"function to create masks of each class given the annotation data and crop(image) 
            coordinates=(4X2 shape) also the donsample factor of the crop to scale the polygon coords"""
        annotation_class=class_annotation_data['class_name'].iloc[0]
        annotation_num=self.class2num[annotation_class]
        
        
        ## select the top left point of the crop
        ## its the point with the min X and Y corrdinates (top left of image is origin)
        
        top_left=crop.min(axis=0)
        
        ## create a shapely polygon from crop to find intersections between annotations and crop
        
        crop_poly=Polygon(crop)
        
        ## create list of intersecting polygons with crop to fill with clss encoding
        
        intersects=[]
        for poly in  class_annotation_data['polygon']:
            if not crop_poly.intersects(poly):
                continue
            else:
                intersect=crop_poly.intersection(poly)
                
                if isinstance(intersect,MultiPolygon):
                    for inter in intersect.geoms:
                        ext_coords=((np.array(inter.convex_hull.exterior.coords)-top_left)//downsample_factor).astype(np.int32)
                        intersects.append(ext_coords)
                elif isinstance(intersect,Polygon):
                        ext_coords=((np.array(intersect.convex_hull.exterior.coords)-top_left)//downsample_factor).astype(np.int32)
                        intersects.append(ext_coords)
                else:
                        continue
                        
                        
                

                    
        mask=np.zeros(self.crop_pixel_size,dtype=np.uint8)
        
       
        ## fill the intersected polygons within the mask
        cv2.fillPoly(mask,intersects,color=annotation_num)
        
        return torch.tensor(mask,dtype=torch.uint8)
        
        
        
    def read_slide_region(self,slide_obj:openslide.OpenSlide,top_left:np.array,
                         level:int):
        """ returns the pixel RGB from WSI given a location,crop_size and level"""
        #pdb.set_trace()
        return slide_obj.read_region(tuple(top_left),level,self.crop_pixel_size)

    
    def __getitem__(self, index):
        ## select annotation 
        anno_row=self.anno_df.iloc[index]
        
        ## select all annotations in the same image as indexed annotation
        image_name,anno_class=anno_row['image_name'],anno_row['class_name']
        dowsample_levels=anno_row['downsample_levels']
        image_path=anno_row['image_path']
        image_anno_data=self.anno_df[self.anno_df['image_name']==image_name]
        
        ## select random crop from the annotation
        crop,zoom_level_idx=self.get_random_crop(anno_row)
        
        ## get the top left corner of the crop
        top_left=crop.min(axis=0).astype(np.int32)
         ## Zoom level idx is a tensor convert to int to use as level in openslide
        zoom_level=zoom_level_idx.item()
        
        ## Zoom level is a tensor convert to int to use as key
        downsample_factor=dowsample_levels[zoom_level]
        
        crop_RGB=self.read_slide_region(openslide.OpenSlide(image_path),
                                        top_left,
                                        zoom_level)
                                        
        
       ## get  classwise masks from the selected crop by selecting all annos on the same image
        ## and within the same crop
        
        get_classwise_masks=partial(self.get_mask_per_class,
                                          crop=crop,
                                          downsample_factor=downsample_factor)
        
        
      
        class_wise_masks=image_anno_data.groupby('class_name').apply(get_classwise_masks)
        
       
        
       
        ## stack the masks of various classes
        stacked_masks=torch.stack(class_wise_masks.to_list(),dim=0)
       
        ## create a composite mask with higher class numbers taking precedence in case of ties
       
        composite_mask=stacked_masks.max(dim=0)
        img_tens=torch.tensor(np.array(crop_RGB))[:,:,:-1].permute(2,0,1)
        
        return img_tens,composite_mask.values,zoom_level_idx
