from data.dataset import Dataset

class JainDataset(Dataset):
	delimiter = '\t'
	url = 'http://cs.joensuu.fi/sipu/datasets/jain.txt'

class SpiralDataset(Dataset):
	delimiter = '\t'
	url = 'http://cs.joensuu.fi/sipu/datasets/spiral.txt'