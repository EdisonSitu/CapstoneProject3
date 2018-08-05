import pickle
import numpy as np

class Database:
	def __init__(self, filename):
		"""Initializes with a filename for the database to store to
		Parameters
		----------
		filename: String
		A String representing the file that it is supposed to store to
		"""
		self.filename = filename;
		self.dictionary = {}
	def put(self, image_id, e_image):
		"""Puts a pair of name and average descriptor into the dictionary"""
		self.dictionary[image_id] = e_image
	def get(self, image_id):
		"""Retrieves descriptor based off name"""
		return self.dictionary[image_id]
	def delete(self, image_id):
		self.dictionary.pop(image_id, None)
	def save(self):
		"""Saves dictionary into file"""
		with open(self.filename, mode="wb") as file:
			pickle.dump(self.dictionary, file)
	def load(self):
		"""Loads dictionary from file"""
		with open(self.filename, mode="rb") as file:
			self.dictionary = pickle.load(file)

def get_avg_descriptor(descriptors):
	"""
	Parameters
	----------
	    descriptors: List
	        List of descriptors for one person
	Returns
	-------
	    avg_descriptor: np array
	        Average descriptor"""
	return np.mean(np.vstack(tuple(descriptors)), axis=0)
