import twitter
import json, pickle, codecs
import pdb

api = twitter.Api(consumer_key='rC6233f0jsP0hxnOU1rlbCeMD',
        consumer_secret='ymqnFg0hNseZx6vh59ZATkggVru9bxdYWW62HANktWZLDcCzj9',
        access_token_key='813448207011233792-poPBcJNzMBuQUKAW0rpszuKwIA8iZ0c',
        access_token_secret='6sCi83cwjbBvbiuIwF6d4aFZ6py9fO205ytQUBon0dHE4')

with open('annotation.tsv') as f:
    tweets = f.readlines()
    tweets = [line.split() for line in tweets]

not_found = []
data = {}

def pass1():
    total = len(tweets)
    for i, (t_id, label) in enumerate(tweets):
        try:
            #status = api.GetStatus('552308801228259328')
            status = api.GetStatus(t_id)
            data[t_id] = {
                    'tweet' : status.text,
                    'label' : label
                    }
            print 'DONE: %d/%d  |   not-found: %d' %(i+1, total, len(not_found))
        except twitter.error.TwitterError as e:
            not_found.append(t_id)
            print 'ERROR:', e.message[0]['message']
            #pdb.set_trace()
    
    print "DONE"
    pdb.set_trace()
    #with codecs.open('data2.txt', 'w', 'utf-8') as f:
    #   json.dump(data, f, ensure_ascii=False)
    #with open('not_found.txt', 'wb') as f:
    #    pickle.dump(not_found, f)

def pass2():        # CONTINUE where we left off!
    with open('not_found.txt', 'rb') as f:
        not_found = pickle.load(f)
    with codecs.open('data2.txt', 'r', 'utf-8') as f:
        data = json.load(f)

    tweet_map_for_labels = dict(tweets)
    for i, t_id in enumerate(not_found):
        try:
            status = api.GetStatus(t_id)
            print 'FOUND: %s, appending to main dict' % (t_id)
            data[t_id] = {
                    'tweet' : status.text,
                    'label' : tweet_map_for_labels[t_id]
                    }
            print 'Done: %d/%d' %(i+1, len(not_found))
        except twitter.error.TwitterError as e:
            print 'ERROR: %s, %s ' %(t_id, e.message[0]['message'])
    print 'DONE !!'
    pdb.set_trace()


if __name__=="__main__":
    #pass1()
    pass2()
