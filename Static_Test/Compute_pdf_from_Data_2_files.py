import numpy as np

class ABC_Data:

	def __init__(self, folder):

		self.folder = folder
		self.production_data = rst_to_array(os.path.join(self.folder, f'test-Production_001.rst'))

	def extract_pdf_P (self):

		production_pdf= pdf_from_array_with_x(self.production_data, 100, [-5, 5])
		production_bins= production_pdf[0]
		production_logpdf = take_safe_log(production_pdf[1])
		np.savetxt('Smagorinsky log pdf', production_logpdf)
		np.savetxt('Smagorinsky bins', production_bins)

	def extract_pdf_Sigma (self):




ABC_Data ('/Users/pablo/Documents/Pablo_Stats/data')
ABC_Data.extract_pdf_P()
ABC_Data.extract_pdf_Sigma()











