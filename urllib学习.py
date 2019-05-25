import urllib.request

url = "http://fs.w.kugou.com/201903091850/547e280b1e8c6139217c5afe5bb31559/G022/M06/12/03/9pMEAFWKLteAftb0AD8A_yIk33g406.mp3"
he = {
    "User-Agnet":"Mozilla/5.0 (Windows NT 6.1; WOW64) "
                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                 "Chrome/57.0.2987.98 Safari/537.36 LBBROWSER"}

req = urllib.request.Request(url,headers=he)
response = urllib.request.urlopen(req)

data = response.read()

with open(r"C:\Users\zhaowei\Desktop\PycharmProjects\风筝误.mp3","wb") as f:
    f.write(data)
