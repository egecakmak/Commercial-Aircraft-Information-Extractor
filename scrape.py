import requests
from bs4 import BeautifulSoup
import bs4

continent_links = ['https://en.wikipedia.org/wiki/List_of_airlines_of_Africa',
                   'https://en.wikipedia.org/wiki/List_of_airlines_of_the_Americas',
                   'https://en.wikipedia.org/wiki/List_of_airlines_of_Asia',
                   'https://en.wikipedia.org/wiki/List_of_airlines_of_Europe',
                   'https://en.wikipedia.org/wiki/List_of_airlines_of_Oceania']

links = []
airlines = set()
# Get all links that contain airline names.
for continent in continent_links:
    html = requests.get(continent)
    b = BeautifulSoup(html.text, 'lxml')
    country_divs = b.find_all('div', {"role": "note"})
    country_links = [each.contents[1].attrs['href'] for each in country_divs]
    links.extend(country_links)

for link in links:
    # Get the content of each link.
    html = requests.get('https://en.wikipedia.org' + link)
    b = BeautifulSoup(html.text, 'lxml')
    # Airline names are stored in tables and sometimes there can be multiple tables. As a result of that, we get all the
    # tables in the page.
    tables = b.find_all('table', {"class": lambda c: "wikitable" in c})
    if len(tables) > 0:
        for table in tables:
            airline_trs = table.contents[1].contents
            # Get the order of the column that contains the airline name.
            table_column_names = table.contents[1].contents[0]
            counter = 0
            for column in table_column_names:
                try:
                    column_text = column.text.strip().upper()
                    if 'AIRLINE' in column_text or 'NAME' in column_text:
                        break
                except AttributeError:
                    pass
                counter += 1
            # For each row get the content of the column that has the airline name and put it in airlines set.
            for tr in airline_trs:
                if isinstance(tr, bs4.element.Tag):
                    airline_column = tr.contents[counter]
                    if airline_column.name == 'td':
                        try:
                            airlines.add(airline_column.contents[0].text.strip())
                        except AttributeError:
                            airlines.add(airline_column.text.strip())
                        except Exception as e:
                            print(e)

# Turns our set to a list and sorts it. Finally, writes its contents in a text file.
airlines = list(airlines)
airlines.sort()
with open('airline_list.txt', 'w') as airline_list:
    for airline in airlines:
        try:
            airline_list.write(airline.strip() + '\n')
        except Exception as e:
            print(e)
            print(airline)
