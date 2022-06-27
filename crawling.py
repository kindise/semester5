import tweepy
import csv

# fungsi untuk mencari sebuah tweet
def caritweet(query):
    # key untuk autentikasi dari twitter developer
    consumer_key = 'Id6KaN5otpbX2aKMy5laeShSm'
    consumer_secret = 'OQhuDqGSO0fLheAs5x0mggKKFOonadJ0u4jjsXtpZdKl74EYxw'
    access_token = '1537022401065943040-GujcVF5mpu0E3IRzB3EER47RGaj5fz'
    access_token_secret = 'X8UlnigGWbPZVBHmtHoE9tEqiU0g0jiLauH4PBDlGLZ66'

    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, 
            access_token, access_token_secret)
    api = tweepy.API(auth)

    # ambil data tweet sebanyak 1000 data berdasarkan tanggal yang ditentukan
    tweets = tweepy.Cursor(method=api.search_30_day,label='kinapp', query= query, fromDate='202206010000',toDate='202206090000',).items(10)
    return tweets

# fungsi untuk export csv
def exportcsv(tweets):
    # header atau title untuk csv
    header = ['id', 'created_at', 'text']
    with open('dataset.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for tweet in tweets:
            writer.writerow([
                tweet.id, tweet.created_at, tweet.text
            ])

if __name__ == '__main__':
    query = '("Shopee Food" OR "ShopeeFood") lang:id -has:links'
    tweets = caritweet(query)
    exportcsv(tweets)