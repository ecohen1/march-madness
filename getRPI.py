from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import csv

urls = {
	'rpi': 'https://www.teamrankings.com/ncb/rpi',
	'sos': 'https://www.teamrankings.com/ncaa-basketball/ranking/schedule-strength-by-other'
}

data = 'sos'
keys = ['season','team',data]
with open('data/processed/'+data+'.csv', 'w') as output_file:
	writer = csv.DictWriter(output_file, keys)
	writer.writeheader()

	years = [ i for i in range(2003,2018) ]

	for yr in years:
		html = urlopen(urls[data]+ '?date=' + str(yr) + '-03-10')
		soup = bs(html, 'lxml')

		table = soup.find('table')
		tdList = table.findChildren('td')
		
		team_ratings = []
		team_dict = {}
		for i,td in enumerate(tdList):
			if i%6 == 1:
				print(td.text)
				if '(' in td.text:	
					team_dict['team'] = '('.join(td.text.split('(')[:-1])
				else:
					team_dict['team'] = td.text.strip()
			elif i%6 == 2:
				team_dict[data] = td.text.strip() if td.text.strip() != '--' else '0.300'
				team_dict['season'] = str(yr)
				team_ratings.append(team_dict)
			else:
				team_dict = {}

		writer.writerows(team_ratings)
