import os
import xlrd
from urllib.request import urlretrieve


excel_path = r'Book1.xlsx'
data = xlrd.open_workbook(excel_path)
table = data.sheets()[0]

nrows = table.nrows
for i in range(1, nrows):
	row = table.row_values(i)
	url_str = row[0]
	if url_str[0] == "'":
		url = url_str[1:-1]
	else:
		url = url_str[0:-1]
	print(url)
	url_arr = url.split('/')
	name_2, name_1 = url_arr[-1], url_arr[-2]
	image_name = name_1 + '_' + name_2
	image_path = os.path.join('emotionet', image_name)
	try:
		urlretrieve(url, image_path) 
	except Exception as e:
		continue
