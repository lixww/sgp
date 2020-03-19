from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from crawler.items import CrawlerItem



class ImagesSpider(CrawlSpider):
    ''' `usage:` scrapy crawl [name] -o [record.json]'''

    name = "images"
    start_urls = [
        'http://www.digitalgalen.net/Data/214v-221r/',
    ]
    rules = (
        Rule(LxmlLinkExtractor(deny_extensions=['md5', 'xmp', 'html']), callback='parse', follow=True),
    )


    def parse(self, response):
        item = CrawlerItem()
        item['jpg_urls'] = []
        linkextractors = LxmlLinkExtractor(
            allow=[r'\.jpg', r'\.tif'], 
            deny_extensions=['md5', 'xmp', 'html']
        )
        for link in linkextractors.extract_links(response):
            item['jpg_urls'].append(link.url)
        return item