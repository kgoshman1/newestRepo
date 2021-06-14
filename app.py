import matplotlib as matplotlib
from flask import Flask, request, render_template, Response
import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from io import StringIO

image1 = None
image2 = None


def fig2data(fig):

    import PIL.Image as Image

    fig.canvas.draw()

    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)


    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w,h), buf.tostring())
    image = np.asarray(image)
    return image

def fig3data(fig):

    import PIL.Image as Image

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image

def defcon1(df):
    global image1, image2, image3

    figure = plt.figure(figsize=(16,8))
    plt.title("Bitcoin")
    plt.xlabel('Days')
    plt.ylabel('Close Price USD')
    plt.plot(df['Close'])
    plt.legend(['Orig', 'Val', 'Pred'])
    image1 = fig2data(figure)

    df = df[['Close']]
    df.head(4)

    future_days = 25

    df['Prediction'] = df[['Close']].shift(-future_days)
    df.tail(4)

    X = np.array(df.drop(['Prediction'], 1))[:-future_days]
    print(X)

    y = np.array(df.drop(['Prediction'],1))[:-future_days]
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    tree = DecisionTreeRegressor().fit(x_train, y_train)

    lr = LinearRegression().fit(x_train, y_train)

    x_future = df.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    print(x_future)

    tree_prediction = tree.predict(x_future)
    print(tree_prediction)
    print()

    lr_prediction = lr.predict(x_future)
    print(lr_prediction)

    predictions = tree_prediction

    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions

    figure1 = plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Orig', 'Val', 'Pred'])
    image2 = fig2data(figure1)

    predictions2 = lr_prediction
    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions2

    figure3 = plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Orig', 'Val', 'Pred'])
    image3 = fig3data(figure3)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("login.html")

database = {'admin': 'admin'}

@app.route('/form_login', methods=['POST', 'GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    if name1 not in database:
        return render_template('login.html', info='Invalid User')
    else:
        if database[name1] != pwd:
            return render_template('login.html', info='Invalid Password')
        else:
            return render_template('uploadFile.html', name=name1)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
   if request.method == 'POST' or request.method == 'GET':
       file = request.files['file']

       s = str(file.read(), 'utf-8')

       data = StringIO(s)
       df = pd.read_csv(data, sep=",")

       defcon1(df)

   return render_template('uploadFile.html', message="upload")

def gen(im):
    ret, buffer = cv2. imencode('.jpg', im)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/img1')
def img1():
    global image1
    return Response(gen(image1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/img2')
def img2():
    global image2
    return Response(gen(image2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/img3')
def img3():
    global image3
    return Response(gen(image3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
