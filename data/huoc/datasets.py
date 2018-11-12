import os
from data.dataset import Dataset, Collection
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class HuocDataset(Dataset):
	delimiter = ','
	url = os.environ.get('HUOC_DATASET_URL')
	labels = ['Genero','Idade','Tempo de Estudos','AD8','MEEM','CDR','TFVS','CYP46A1','APOE4','Diagnostico']

class ScaledHuocDataset(HuocDataset):
	scaler = MinMaxScaler

class StdScaledHuocDataset(HuocDataset):
	scaler = StandardScaler

class CognitiveTestsDataset(HuocDataset):
	features = [3, 4, 5, 6, -1]

class ScaledCognitiveTestsDataset(CognitiveTestsDataset):
	scaler = MinMaxScaler

class StdScaledCognitiveTestsDataset(CognitiveTestsDataset):
	scaler = StandardScaler

class CognitivePlusGenderDataset(HuocDataset):
	features = [0, 3, 4, 5, 6, -1]

class ScaledCognitivePlusGenderDataset(CognitivePlusGenderDataset):
	scaler = MinMaxScaler

class StdScaledCognitivePlusGenderDataset(CognitivePlusGenderDataset):
	scaler = StandardScaler


class SocialInfoDataset(HuocDataset):
	features = [0, 1, 2, -1]

class ScaledSocialInfoDataset(SocialInfoDataset):
	scaler = MinMaxScaler

class StdScaledSocialInfoDataset(SocialInfoDataset):
	scaler = StandardScaler

class SocialAndTestsDataset(HuocDataset):
	features = [0, 1, 2, 3, 4, 5, 6, -1]

class ScaledSocialAndTestsDataset(SocialAndTestsDataset):
	scaler = MinMaxScaler

class StdScaledSocialAndTestsDataset(SocialAndTestsDataset):
	scaler = StandardScaler

class GeneticMarkersDataset(HuocDataset):
	features = [7, 8, -1]

class ScaledGeneticMarkersDataset(GeneticMarkersDataset):
	scaler = MinMaxScaler

class StdScaledGeneticMarkersDataset(GeneticMarkersDataset):
	scaler = StandardScaler

class GeneticAndSocialDataset(HuocDataset):
	features = [0, 1, 2, 7, 8, -1]

class ScaledGeneticAndSocialDataset(GeneticAndSocialDataset):
	scaler = MinMaxScaler

class StdScaledGeneticAndSocialDataset(GeneticAndSocialDataset):
	scaler = StandardScaler

class HuocCollection(Collection):
	base = HuocDataset
