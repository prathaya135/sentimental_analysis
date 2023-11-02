

file = open('sample.txt', 'r')
message=0
message1=0
count = 0
message=file.readline()
message1= file.readline()
line=[]
line1=[]
file1 = open('text.txt', 'r')
file2 = open('positive.txt', 'r')
for i in range(6): 
    line.append(file1.readline())

for i in range(6):
    line1.append(file2.readline())

print(message1,message)
message = int(message)
message1 = int(message1)
total = message+message1
message = (message/total)*100
message1 = (message1/total)*100

message = round(message)
message1= round(message1)
from flask import Flask, render_template

app = Flask(__name__)

# Sample list of items
item =[message,message1]
@app.route('/second')
def index():
    return render_template("second.html",message=message,message1=message1,message2=line[0],message3=line[1],message4=line[2],message5=line[3],message6=line[4],message7=line1[0],message8=line1[1],message9=line1[2],message10=line1[3],message11=line1[4])
    # return "hello world"

if __name__ == '__main__':
    app.run(debug=True)