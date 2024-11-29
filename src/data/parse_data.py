from bs4 import BeautifulSoup, element
import requests

# parse specific site
def parse_site(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    category_list = soup.find_all('div', {'class': 'category-list'})
    category_links = category_list[0].find_all('a')
    
    full_text = ''
    for category_link in category_links:
        r = requests.get(category_link['href'])
        soup = BeautifulSoup(r.content, 'html.parser')
        
        pagination_holder_links = soup.find_all('a', {'class': 'pagination-holder-link'})
        
        category_text = ''
        for pagination_holder_link in pagination_holder_links[:len(pagination_holder_links)-1]:
            r = requests.get(pagination_holder_link['href'])
            soup = BeautifulSoup(r.content, 'html.parser')
        
            content_block_section = soup.find_all('div', {'class': 'content-block section'})
            items = content_block_section[0].find_all('li')
            page_text = ''
            for item in items:
                if item.p is not None: 
                    contents = item.p.contents
                    contents_text = ''
                    for content in contents:
                        if type(content) == element.NavigableString:
                            contents_text += str(content)
                        elif content.name == 'br':
                            contents_text += '\n'
                        elif type(content) == element.Tag:
                            contents_text += str(content.string)
                    page_text += contents_text + '\n\n'
            category_text += page_text + '\n\n'
        full_text += category_text + '\n\n'
        
    with open('../../data/anecdotes1.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)
    
url = ''
parse_site(url)