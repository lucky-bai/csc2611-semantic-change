# Try out some Classical Chinese POS taggers on a sample 24hist text
# MeCab (trained on Kyoto corpus)
# UDPipe (trained on Kyoto corpus)
# Stanford CoreNLP (modern Chinese)
import udkanbun
import stanfordnlp

TEXT = """
李白，字太白，山東人。少有逸才，志氣宏放，飄然有超世之心，父為任城尉，因家焉。少與魯中諸生孔巢父、韓凖、裴政、張叔明、陶沔等隠於徂徠山，酣歌縱酒，時號「竹溪六逸」。天寳初，客遊㑹稽，與道士吴筠隠於剡中。既嗜酒，日與飲徒醉於酒肆。𤣥宗度曲，欲造樂府新詞，亟召白，白已臥於酒肆矣。召入，以水灑面，即令秉筆，頃之成十餘章，帝頗嘉之。甞沉醉殿上，引足令髙力士脱靴，由是斥去。乃浪迹江湖，終日沉飲。時侍御史崔宗之謫官金陵，與白詩酒唱和，甞月夜乘舟，自采石逹金陵，白衣宫錦袍，於舟中顧瞻笑傲，傍若無人。初，賀知章見白，賞之曰：「此天上謫仙人也。」禄山之亂，𤣥宗幸蜀，在途以永王璘為江淮兵馬都督、揚州節度大使。白在宣州謁見，遂辟從事。永王謀亂，兵敗，白坐長流夜郎。後遇赦得還，竟以飲酒過度，醉死於宣城。有文集二十卷，行於時。
"""

# MeCab tagger
# UDPipe: set MeCab=False
lzh = udkanbun.load(MeCab=False)
s = lzh(TEXT)
print(s)

# Stanford CoreNLP tagger (modern Chinese)
#nlp = stanfordnlp.Pipeline(lang="zh")
#doc = nlp(TEXT)
#for sent in doc.sentences:
#  for w in sent.words:
#    print(w.text, w.pos)

