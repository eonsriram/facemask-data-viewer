from flask import Flask, render_template, request, redirect, url_for
import random
import string
from sqlops import SQLDB


#db = SQLDB()
application = Flask(__name__)

def generator(size=7, chars= string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@application.route('/')
@application.route('/login')
def index():
    return render_template('login.html')


@application.route('/process', methods=['POST'])
def process():
    user = request.form['user']
    password = request.form['pass']

    db = SQLDB()
    cred = db.read("SELECT * FROM nie.credentials WHERE user = '{}';".format(user))
    db.close()

    if len(cred) == 0:
        return render_template('login.html', data="Invalid User")
    elif password == cred[0][1]:
        return redirect(url_for('table'))
    else:
        return render_template('login.html', data="Wrong Password")


@application.route('/table')
@application.route(f'/{generator()}')
def table():
    db = SQLDB()
    data = db.read("SELECT * FROM nie.log;")
    db.close()
    return render_template('table.html', data=data)


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0')
