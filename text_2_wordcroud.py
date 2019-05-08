'''ここにテキストを入力してワードクラウドを生成するようにしたい'''
#テキストの挿入
from rnnlm_gen import RnnlmGen
from preprocess import preprocess

#文章の読み込み
path_w = 'items/10. Survey.txt'
with open(path_w, mode='r',encoding="utf-8_sig") as f:
    text = f.read()
    
#text = "Consciousness and its background of infringement of others right 1 background With the development of information and communication technology, it has become an information society and peoples lives are also diversified. Along with this, new crimes using the Internet and digital devices are also increasing. Also, on the other hand, the conventional crimes are on the decline. The number of perceived criminal law criminals has been consistently decreasing since the peak in 2002, and there has been a certain improvement in the crime situation. However, in addition to child abuse, stalking cases, and cases of violence from spouses tending to increase, the total damage of special fraud including payment fraud in 27 years will be about 48.2 billion yen. The situation is still unpredictable. In addition, cyber threats are frequent and cyber attacks are continuing, and threats in cyber space are becoming serious.2016 version police white paper p. 64 As a whole, while acknowledging the improvement of the criminal situation, on the other hand, the increase in child abuse, stalking cases, violent cases from spouses, special fraud including transfer fraud etc. It mentions the growing seriousness of cybercrime. Next, we will look at the transition of cybercrime. There is a description in Section 1 The current state of cybercrime in  Feature II: Aiming for the realization of a safe, secure and responsible cyber civil society. 1 General arrest situation The number of arrests for cybercrime continues to increase, reaching 6,933 in 2010, an increase of 243 cases 3.6 over the previous year, the highest ever. In addition, the number of arrests for network use crimes during 22 years also reached 5,199, an increase of 1,238 31.3 over the previous year, reaching a record high.(Feature II: Aiming for the realization of a safe, secure and responsible cyber civil society H13 ~ 22 In addition, the 2016 Police White Paper also includes the transition from 2011 to 2011, “Chapter 3 Securing Cyberspace Safety, Section 1: Threats to Cyberspace, Arrest of Cybercrime In the Situation there is the following description. The number of arrests for cybercrime during 2015 increased by 191 from the previous year to 8,096. The number of arrests for violation of the unauthorized access law increased by 9 cases from the previous year to 373 cases. The number of arrested persons increased by 3 to 173, compared with the previous year. The number of arrests for crimes relating to computer and electromagnetic records prescribed in the Penal Code and for crimes relating to fraudulent instructions increased by 48 from the previous year to 240 Of these, 45 cases were found for computer virus crimes. The number of arrests for networked crimes increased by 134 to 7,483, compared with the previous year.As mentioned above, “The number of perceived criminal offenses has consistently decreased since the peak of 2002, and there has been a certain improvement in the crime situation”  However, it can be seen that cybercrime specific to the information society has been increasing steadily in recent years. The Japanese government is not only involved in the growing number of cybercrime, but is also responding in various ways. One of them is It is illegal because of the revision of the Copyright Act enforced on January 1, 2010.Downloading while knowing that music video by Internet distribution is illegal is infringing violation of copyright law even for private use purposes."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
corpus_size = len(corpus)
model = RnnlmGen()
start_word = 'law'
start_id = word_to_id[start_word]
skip_words = []
skip_ids = [word_to_id[w] for w in skip_words]
# 文章生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
#文章を保存
path_w = 'items/10. Survey2.txt'
with open(path_w, mode='w', errors='ignore') as f:#UnicodeEncodeError: 'cp932' codec can't encode character '\u200b' in position 1274: illegal multibyte sequence
    f.write(txt)

#使用するモジュールをimport
import wordcloud, codecs
#テキストファイルを読み込み
file = codecs.open(path_w, 'r', 'utf-8_sig', errors='ignore')#UnicodeDecodeError: 'utf-8' codec can't decode byte 0x81 in position 2106: invalid start byte
text = file.read()
#テキストからwordcloudを生成
wordc = wordcloud.WordCloud(background_color='white', width=800,height=600).generate(text)
#画像ファイルとして保存
wordc.to_file('items/10. Survey2.png')