import os
from data.dataset import Dataset, Collection
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class HuocDataset(Dataset):
	delimiter = ','
	name = 'Full arrangement'
	url = os.environ.get('HUOC_DATASET_URL')
	labels = ['Genero','Idade','Tempo de Estudos','AD8','MEEM','CDR','TFVS','CYP46A1','APOE4','Diagnostico']

class ScaledHuocDataset(HuocDataset):
	name = 'Scaled Full arrangement'
	scaler = MinMaxScaler

class StdScaledHuocDataset(HuocDataset):
	name = 'Standard Full arrangement'
	scaler = StandardScaler

class CognitiveTestsDataset(HuocDataset):
	name = 'Only cognitive tests'
	features = [3, 4, 5, 6, -1]

class ScaledCognitiveTestsDataset(CognitiveTestsDataset):
	name = 'Scaled cognitive tests'
	scaler = MinMaxScaler

class StdScaledCognitiveTestsDataset(CognitiveTestsDataset):
	name = 'Standard cognitive tests'
	scaler = StandardScaler

class CognitivePlusGenderDataset(HuocDataset):
	name = 'Cognitive tests plus gender'
	features = [0, 3, 4, 5, 6, -1]

class ScaledCognitivePlusGenderDataset(CognitivePlusGenderDataset):
	name = 'Scaled cognitive tests plus gender'
	scaler = MinMaxScaler

class StdScaledCognitivePlusGenderDataset(CognitivePlusGenderDataset):
	name = 'Standard cognitive tests plus gender'
	scaler = StandardScaler

class SocialInfoDataset(HuocDataset):
	name = 'Only social information'
	features = [0, 1, 2, -1]

class ScaledSocialInfoDataset(SocialInfoDataset):
	name = 'Scaled social information'
	scaler = MinMaxScaler

class StdScaledSocialInfoDataset(SocialInfoDataset):
	name = 'Standard social information'
	scaler = StandardScaler

class SocialAndTestsDataset(HuocDataset):
	name = 'Social information plus cognitive tests'
	features = [0, 1, 2, 3, 4, 5, 6, -1]

class ScaledSocialAndTestsDataset(SocialAndTestsDataset):
	name = 'Scaled social information plus cognitive tests'
	scaler = MinMaxScaler

class StdScaledSocialAndTestsDataset(SocialAndTestsDataset):
	name = 'Standard social information plus cognitive tests'
	scaler = StandardScaler

class GeneticMarkersDataset(HuocDataset):
	name = 'Only genetic markers'
	features = [7, 8, -1]

class ScaledGeneticMarkersDataset(GeneticMarkersDataset):
	name = 'Scaled genetic markers'
	scaler = MinMaxScaler

class StdScaledGeneticMarkersDataset(GeneticMarkersDataset):
	name = 'Standard genetic markers'
	scaler = StandardScaler

class GeneticAndSocialDataset(HuocDataset):
	name = 'Genetic markers plus social information'
	features = [0, 1, 2, 7, 8, -1]

class ScaledGeneticAndSocialDataset(GeneticAndSocialDataset):
	name = 'Scaled genetic markers plus social information'
	scaler = MinMaxScaler

class StdScaledGeneticAndSocialDataset(GeneticAndSocialDataset):
	name = 'Standard genetic markers plus social information'
	scaler = StandardScaler

class HuocCollection(Collection):
	base = HuocDataset
