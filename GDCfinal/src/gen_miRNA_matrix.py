# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os,pdb
from utils import logger
def file_as_bytes(file):
    with file:
        return file.read()

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id

	'''
	count = 0

	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find("-") != -1:

					filepath = idpath + "/" + filename
					df = pd.read_csv(filepath,sep="\t")
					# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df.miRNA_ID.values.tolist()

					id_miRNA_read_counts = [idname] + df.read_count.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)


					count +=1
					# print (df)
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df

def extractLabel(inputfile):
	df = pd.read_csv(inputfile, sep="\t")
	#
	# print (df[columns])
	#pdb.set_trace()
	df['label'] = df['cases.0.project.primary_site']+' '+df['cases.0.samples.0.sample_type']
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = 'Normal'
	label =1;
	primary_type = {}
	for i,data in enumerate(df['cases.0.project.primary_site']):
		
			if(df['cases.0.samples.0.sample_type'][i].find("Normal") >= 0):
					df['label'][i] = 0
			else:
				try:
					df['label'][i] = primary_type[data]
				except KeyError:
					primary_type[data] = label
					df['label'][i] = label
					label = label+1
					
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = 0
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Primary Tumor"), 'label'] = 1
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("New Primary"), 'label'] = 2
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Additional Metastatic"), 'label'] = 3
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Control"), 'label'] = 4
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Metastatic"), 'label'] = 5
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Primary Blood Derived Cancer - Bone Marrow"), 'label'] = 6
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Primary Blood Derived Cancer - Peripheral Blood"), 'label'] = 7
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Recurrent Blood Derived Cancer - Bone Marrow"), 'label'] = 8
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Recurrent Blood Derived Cancer - Peripheral Blood"), 'label'] = 9
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Recurrent Tumor"), 'label'] = 10
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Cell Lines"), 'label'] = 11
	
	#tumor_count = df.loc[df.label == 1].shape[0]
	#normal_count = df.loc[df.label == 0].shape[0]
	#logger.info("{} Normal samples, {} Tumor samples ".format(normal_count,tumor_count))
	columns = ['file_id','label']
	return df[columns]

if __name__ == '__main__':


	data_dir ="/Users/lenovo/Downloads/GDCfinal/data_new/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	dirname = data_dir + "live_miRNA"
	label_file = data_dir + "files_meta.tsv"
	
	#output file
	outputfile = data_dir + "miRNA_matrix.csv"

	# extract data
	matrix_df = extractMatrix(dirname)
	label_df = extractLabel(label_file)

	#merge the two based on the file_id
	result = pd.merge(matrix_df, label_df, on='file_id', how="left")
	#print(result)

	#save data
	result.to_csv(outputfile, index=False)
	#print (labeldf)

 




