from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm




class WSIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "WSI_download",
                     GCP_auth_file:str="eighth-sensor-386109-5d86556be43b.json",
                    bucket_name='wsi-images'
                       ):
        """download WSI data to data_dir from GCP. requires GCP credentials: GCP_auth_file
            and bucket name as created on GCP """
        
        super().__init__()
        
        ## create folder to save gcp files
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.GCP_auth_file=GCP_auth_file
        self.bucket_name=bucket_name
       
    
    # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    
    
    ## define Google Cloud Platform storage client-to download WSI image data from the cloud
    
    def get_storage_bucket(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']=self.GCP_auth_file
        storage_client=storage.Client()
        return storage_client.get_bucket(self.bucket_name)
    
    def prepare_data(self,file_types:list=['.tiff','.JPG']):
        """download data from GCP  arranged in the same folder structure as GCP.
        specify filetypes/suffixes in the file_types param"""
        
        # download
        storage_bucket=self.get_storage_bucket()
        
        for blob in tqdm(list(storage_bucket.list_blobs())):
            destination_filename=self.data_dir/blob.name
             ## create empty file with destination filename to fill with downloaded blob
                
            if destination_filename.suffix in file_types:
                dest_path=destination_filename.parent
                dest_path.mkdir(exist_ok=True)
                
                with open(destination_filename, 'w') as fp:
                    pass
                blob.download_to_filename(str(destination_filename))

                    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
