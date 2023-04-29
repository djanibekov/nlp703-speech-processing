import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://www.ocf.berkeley.edu/~acowen/music.html#'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

canvas_all = soup.find_all('div', attrs={'class': 'canvas', 'id':'allratings'})
canvas_usa = soup.find_all('div', attrs={'class': 'canvas', 'id':'USA'})
canvas_chn = soup.find_all('div', attrs={'class': 'canvas', 'id':'CHN'})

dataframe1 = pd.DataFrame(columns=['id', 'ondblclick', 'onmouseover', 'class'])
dataframe2 = pd.DataFrame(columns=['id', 'ondblclick', 'onmouseover', 'class'])
dataframe3 = pd.DataFrame(columns=['id', 'ondblclick', 'onmouseover', 'class'])

counter = 0
for grid, dataframe, name in zip([canvas_all[0], canvas_usa[0], canvas_chn[0]], [dataframe1, dataframe2, dataframe3], ['all', 'usa', 'china']):
    genre_scanme = grid.find_all('div', attrs={'class': 'genre scanme'})
    data = {}
    for sample in genre_scanme:
        class_ = sample.get_text()
        id_ = sample['id']
        ondblclick_ = sample['ondblclick']
        onmouseover_ = sample['onmouseover']
        data['id'] = id_
        data['ondblclick'] = ondblclick_
        data['onmouseover'] = onmouseover_
        data['class'] = class_

        dataframe = pd.concat([dataframe, pd.DataFrame(data, index=[counter])])
        counter = counter + 1
    dataframe.to_csv(f'{name}_music_genre_wo_text.csv', index=False)
