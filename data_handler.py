import json
import pdb
import codecs
import pdb

def get_data():
    tweets = []
    files = ['racism.json', 'neither.json', 'sexism.json']
    for file in files:
        with codecs.open('./tweet_data/' + file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        for line in data:
            tweet_full = json.loads(line)
            tweets.append({
                'id': tweet_full['id'],
                'text': tweet_full['text'].lower(),
                'label': tweet_full['Annotation'],
                'name': tweet_full['user']['name'].split()[0]
                })

    #pdb.set_trace()
    return tweets


if __name__=="__main__":
    tweets = get_data()
    males, females = {}, {}
    with open('./tweet_data/males.txt') as f:
        males = set([w.strip() for w in f.readlines()])
    with open('./tweet_data/females.txt') as f:
        females = set([w.strip() for w in f.readlines()])

    males_c, females_c, not_found = 0, 0, 0
    for t in tweets:
        if t['name'] in males:
            males_c += 1
        elif t['name'] in females:
            females_c += 1
        else:
            not_found += 1
    print males_c, females_c, not_found
    pdb.set_trace()
