import requests


def create_url(query):
    base_url = "https://api.qwant.com/v3/search/images?locale=en_US&q="
    formatted_query = requests.utils.quote(query.replace(" ", "+"))
    rest_of_url = "&t=images&size=large&count=50&offset=0&device=desktop&safesearch=1"
    return base_url + formatted_query + rest_of_url


def fetch_images(query):
    url = create_url(query)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    image_urls = [item['media'] for item in data['data']['result']['items']]

    return image_urls[:1]