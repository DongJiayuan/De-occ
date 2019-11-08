from bs4 import BeautifulSoup
import requests
import os


os.makedirs('.\\occ\\', exist_ok=True)
keyword = ['apple','sunglasses','bear','bread','hand','scarf','hair',
           'book','hat','cabinet','cup','computer','phone','beverage',
           'strawberry','watermelon','pear','caffe','beer','chocolate',
           'cake','candy','food','drink','beard','egg','milk','soup',
           'tea','water','wine','yogurt','ice%20cream','poster','peach',
           'strawberry','watermelon','kfc','hamburger','lipstick','pen',
           'microphone','finger','glasses','toy','bag','PC','popcorn']

URL = "http://www.stickpng.com/search"

for i in keyword:
    os.makedirs('.\\occ\\'+ i +'\\', exist_ok=True)
    for page in range(1,10):
        param ={'q': i,
                'page':str(page)}
#       html = requests.get(URL,params=param,proxies = {"http": "http://127.0.0.1:12333"}).text
        html = requests.get(URL,params=param).text
        soup = BeautifulSoup(html, 'lxml')
        img_ul = soup.find_all('div',{'class':'item'})
        print(img_ul)

        for ul in img_ul:
            print(ul)
            imgs = ul.find_all('img')
            for img in imgs:
                url = img['src']
                r = requests.get(url, stream=True)
                image_name = url.split('/')[-1]
                with open('.\\occ\\'+i+'/%s' % image_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)
                print('Saved %s' % image_name)