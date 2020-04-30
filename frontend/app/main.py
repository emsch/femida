#!/usr/bin/env python3
import os
import uuid
import datetime
import time
import json
import random
import errno
import logging

from flask import (
    Flask, request,
    send_from_directory,
    render_template,
    redirect, url_for,
    flash, Response,
    jsonify, session,
    stream_with_context
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user,
    fresh_login_required,
    current_user
)
from flask_oauthlib.client import OAuth
import bson
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename

# directories are created in prestart.sh file
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/media/pdf_uploads/')
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', '/media/ocr_results/')
MONGO_HOST = os.environ.get('MONGO_HOST', 'localhost')
ALLOWED_EXTENSIONS = {'pdf'}

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_TYPE'] = 'memcached'
app.config["MONGO_URI"] = f"mongodb://{MONGO_HOST}:27017/femida"
app.config['SESSION_PROTECTION'] = 'strong'
app.secret_key = os.environ['FEMIDA_SECRET_KEY']
app.debug = os.environ.get('FEMIDA_DEBUG', False)


# For OAUTH, theese keys must be moved to ENV and reissued
app.config['GOOGLE_ID'] = os.environ.get("GOOGLE_ID")
app.config['GOOGLE_SECRET'] = os.environ.get("GOOGLE_SECRET")
oauth = OAuth(app)

from database import mongo  # noqa
mongo.init_app(app)
pdfs = mongo.db.pdfs
answers = mongo.db.answers
leaderboard = mongo.db.leaderboard

# flask-login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


google = oauth.remote_app(
    'google',
    consumer_key=app.config.get('GOOGLE_ID'),
    consumer_secret=app.config.get('GOOGLE_SECRET'),
    request_token_params={
        'scope': 'email'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)


# silly user model
class User(UserMixin):

    def __init__(self, id, email=None, picture=None):
        self.id = id
        self.name = email
        self.password = self.name + "_secret"
        self.email = email
        self.picture = picture
        self.session_id = id #this will be deprecated

    def __repr__(self):
        return "%d/%s/%s" % (self.id, self.name, self.password)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_runtime_settings():
    cursor = mongo.db.runtime_settings.find()
    try:
        settings = cursor.next()
    except StopIteration:
        mongo.db.runtime_settings.insert_one({
            # Минимальное число ручных проверок (перекрытие)
            'hand_checks': os.environ.get('FEMIDA_HAND_CHECKS', 2),
            'hand_checks_gap': os.environ.get('FEMIDA_HAND_CHECKS_GAP', 10),
            'len_of_audience': list(map(int, os.environ.get('FEMIDA_LEN_OF_AUDIENCE', '3,3').strip().split(','))),
            'names_database': '',
        })
        settings = read_runtime_settings()
    return settings


def update_runtime_settings(**kwargs):
    id_ = read_runtime_settings()['_id']
    mongo.db.runtime_settings.update_one(
        {'_id': id_},
        {'$set': kwargs},
    )


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('img', path)


@app.route(f'/media/<path:path>')
def send_ocr_img(path):
    return send_from_directory(RESULTS_FOLDER, path)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/leaderboard.html')
@fresh_login_required
def serve_leaderboard():
    users = leaderboard.find({}, {'picture': 1, 'email': 1, 'num_of_checks': 1})
    users = [[user['picture'], user['email'], user['num_of_checks']] for user in users]
    users = sorted(users, key=lambda x: x[2], reverse=True)
    all_users_num_of_checks = [user[2] for user in users]
    users = {user[1]: user for user in users} # {email: [picture, email, num_of_checks]}

    places = list(set(all_users_num_of_checks))[::-1] # num_of_checks for every user
    places = {places[i]: i+1 for i in range(len(places))} # {num_of_checks: place}

    params = {}
    params['leaderboard_data'] = [[places[user[2]], *user] for user in users.values()] # [place, picture, email, num_of_checks]
    params['total'] = sum(all_users_num_of_checks)
    params['cur_user_num_of_checks'] = users[current_user.email][2]

    return render_template('leaderboard.html', params=params)

@app.route('/form.html')
@app.route('/index.html')
@app.route('/')
@fresh_login_required
def serve_form():
    candidate_id = request.args.get('id', None)
    if candidate_id is None:
        candidates = answers.find(
            {'$and': [
                {'$where': 'this.manual_checks.length <= %s' % read_runtime_settings()['hand_checks']},
                {'manual_checks': {'$ne': current_user.get_id()}},
                {'status': 'normal'},
                {'$where': 'this.requested_manual.length == 0'},
            ]}
        )
    else:
        flash('Pulling by id=%s' % candidate_id)
        try:
            candidates = answers.find(
                {'_id': ObjectId(candidate_id)},
            )
        except bson.errors.InvalidId as e:
            flash(str(e))
            return render_template('form.html', no_more_candidates=True)

    num_candidates = candidates.count()
    K = int(read_runtime_settings()['hand_checks_gap'])
    HK = (hash(current_user.get_id()) ^ num_candidates) % K
    last = None
    for candidate in candidates:
        last = candidate
        RK = (hash(candidate['_id']) ^ num_candidates) % K
        if RK == HK:
            break

    candidate = last
    if candidate is None:
        return render_template('form.html', no_more_candidates=True)

    # Prepare updates that we already submitted by others
    if len(candidate['test_updates']) > 0:
        updates = list(candidate['test_updates'][-1]['updates'].items())
    else:
        updates = []
    updates += [['', '']] * 12
    updates = [[i+1, str(v), str(o)] for i, (v, o) in enumerate(updates)]
    
    # Prepare fio suggest
    if len(candidate.get('personal', [])) > 0:
        personal = candidate['personal'][-1]
    else:
        personal = dict()

    params = {
        "img_fio": candidate['img_fio'],
        "img_test_form": candidate['img_test_form'],
        "id": candidate['_id'],
        "updates": updates,
        'num_candidates': num_candidates,
        **personal,
    }

    return render_template('form.html', params=params, no_more_candidates=False)


@app.route('/pdf.html')
@fresh_login_required
def serve_pdf():
    len_of_audience = read_runtime_settings()['len_of_audience']
    pattern = "([Пп][0-9]|[0-9]{%d,%d})_([89]|10|11|)_(ОТ|МАТ)_[0-9]{3,5}" % tuple(len_of_audience)
    params = {'pattern': pattern}
    return render_template('pdf.html', params=params)


def valid_form(form):
    class_ = form['class']
    variant = form['variant']
    if class_ and not (class_.isdigit() or class_ == "-"):
        flash(u'''ОШИБКА: некорректный класс. Должно быть число''')
        return False
    if variant and not (variant.isdigit() or variant == '-'):
        flash(u'''ОШИБКА: некорректный вариант. Должно быть число''')
        return False
    for i in range(1, 20):
        val = form.get('fix_q_'+str(i), None)
        if val and not val.isdigit():
            flash(u'''ОШИБКА: некорректное исправление.
                    Должно быть заполнено числом''')
            return False
    return True


def process_updates(form, date, session_id):
    updates = {}
    for i in range(1, 20):
        q = form.get('fix_q_'+str(i), None)
        ans = form.get('fix_ans_'+str(i), None)
        if q and q.isdigit() and ans and ans != '-':
            q = q.lstrip('0')
            updates[q] = ans
    test_updates = {
        'updates': updates,
        'date': date,
        'session_id': session_id
    }
    return test_updates


@app.route('/process_form', methods=['POST'])
def handle_data():
    form = dict(request.form.items())
    if not valid_form(form):
        flash(u'ОШИБКА: неверное расширение у файла или название')
        return redirect(url_for('serve_form'))

    date = datetime.datetime.utcnow()
    session_id = current_user.get_id()
    answer_id = form['id']

    personal = {
        "class": form['class'], "name": form['name'],
        "surname": form['surname'], "patronymic": form['patronymic'],
        "variant": form['variant'],
        "type": form.get('type', ''),
        'session_id': session_id, "date": date,
    }
    if form['status'] == 'manual':
        requested_manual = {'session_id': session_id, "date": date, 
                            'comment': form.get('message-text', "")}
    else:
        requested_manual = None
        updated_leaderboard__id = leaderboard.update(
            {"email": current_user.name},
            {"$set": {"UUID": current_user.name,
                    "picture": current_user.picture,
                    "email": current_user.name},
            "$inc": {"num_of_checks": 1}},
            upsert=True
        )

    test_updates = process_updates(form, date, session_id)

    to_bd = {
        'personal': personal,
        'manual_checks': session_id,
        'test_updates': test_updates
    }

    if requested_manual:
        to_bd['requested_manual'] = requested_manual

    updated_id = answers.update_one(
        {'_id': ObjectId(answer_id)},
        {'$push': to_bd},
    )
    flash('Updated successfully, %s %s' % (updated_id.raw_result, updated_id.upserted_id))

    # flash(jsonify(dict(form=form, personal=personal, updates=updates, requested_manual=requested_manual)))

    # When ready for production
    return redirect(url_for('serve_form'), code=303)


@app.route('/process_pdf', methods=['POST'])
def handle_pdf():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash(u'ОШИБКА: нет файла')
        return redirect(url_for('serve_pdf'))
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash(u'ОШИБКА: нет файла')
        return redirect(url_for('serve_pdf'))
    if not file or not allowed_file(file.filename):
        flash(u'ОШИБКА: неверное расширение у файла или название')
        return redirect(url_for('serve_pdf'))
    uid = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    flash(u'Filename=%s' % filename)
    flash(u'uid=%s' % uid)
    pdf_comment = request.form['pdf_comment']
    flash(u'pdf_comment=%s' % pdf_comment)
    path = os.path.join(app.config['UPLOAD_FOLDER'], uid+"__"+filename)
    file.save(path)
    flash(u'Загрузка файла успешно осуществлена!')
    flash(u'------------------------------------')

    inserted = pdfs.insert_one({
        "UUID": uid,
        "path": path,
        "status": 'waiting for OCR',
        "pdf_comment": pdf_comment,
        "date": datetime.datetime.utcnow(),
        "session_id": current_user.get_id()
    })
    flash(u'Задача на расшифровку поставлена. %s' % inserted.inserted_id)
    flash(u'------------------------------------')

    # Секцию про дергание АПИ с названием файла вставить сюда

    return redirect(url_for('serve_pdf'))


@app.route('/login')
def login():
    session.pop('google_token', None)
    return google.authorize(callback=url_for('authorized', _external=True))


@app.route('/login/authorized')
def authorized():
    resp = google.authorized_response()
    session['google_token'] = (resp['access_token'], '')
    me = google.get('userinfo')
    flash(str(me.data))
    user = User(str(uuid.uuid1())[:8], me.data["email"], me.data["picture"])
    login_user(user, remember=True)

    #return jsonify({"data": session.get('google_token'), "resp": resp, "medata": me.data})
    return redirect("../../")


# handle login failed
@app.errorhandler(401)
def page_not_found(e):
    return Response('<p>Login failed</p>')


# callback to reload the user object
@login_manager.user_loader
def load_user(userid):
    # this will logout user if he logouts his Google account
    try:
        me = google.get('userinfo')
        return User(userid, me.data['email'], me.data['picture'])
    except Exception as e:
        return None



@stream_with_context
def event_stream():
    for i in range(1):
        num = random.randint(0, 20)
        pdf_statuses = {r['_id']: r['count'] for r in pdfs.aggregate(
            [{"$group": {'_id': "$status", 'count': {'$sum': 1}}}])}
        message = {
            'text': 'hey',
            'number': num,
            'pdf_statuses': pdf_statuses
        }
        yield 'data: %s\n\n' % json.dumps(message)
        time.sleep(1)


@app.route("/manager_flow")
@fresh_login_required
def manager_flow():
    return Response(event_stream(), mimetype="text/event-stream")


COLUMNS = [
      {
        "field": "comment",  # which is the field's name of data key
        "title": "Пачка",  # display as the table header's name
        "sortable": True,
      },
      {
        "field": "num_works",
        "title": "Число работ",
        "sortable": True,
      },
      {
        "field": "num_checks",
        "title": "Число проверок",
        "sortable": True,
      },
      {
        "field": "min_checks",
        "title": "Минимум проверок",
        "sortable": True,
      },
      {
        "field": "max_checks",
        "title": "Максимум проверок",
        "sortable": True,
      },
      {
        "field": "num_requested_manual_checks",
        "title": "Запрошено ручных проверок",
        "sortable": True,
      },
    ]


@app.route('/monitor.html', methods=["GET", "POST"])
@fresh_login_required
def serve_monitor():
    if request.method == 'POST':
        form = dict(request.form.items())
        update_runtime_settings(names_database=form.get('fio_field', ''))
        return redirect(url_for('serve_monitor'))

    else:
        data = list(answers.aggregate([
            {'$lookup': {
                'from': 'pdfs', 'localField': 'UUID', 'foreignField': 'UUID', 'as': 'pdf_info'
            }},
            {'$unwind': '$pdf_info'},
            {'$group': {
                '_id': {'comment': '$pdf_info.pdf_comment'},
                'num_works': {'$sum': 1},
                'num_checks': {'$sum': {"$size": '$manual_checks'}},
                'min_checks': {'$min': {"$size": '$manual_checks'}},
                'max_checks': {'$max': {"$size": '$manual_checks'}},
                'num_requested_manual_checks': {'$sum': {"$size": '$requested_manual'}},
            }}
        ]))
        for row in data:
            row['comment'] = row['_id']['comment']

        # other column settings -> http://bootstrap-table.wenzhixin.net.cn/documentation/#column-options
        fios = read_runtime_settings().get('names_database', "")
        return render_template('monitor.html', table_data=data, table_columns=COLUMNS, fios=fios)


@app.route("/get_db.json")
@fresh_login_required
def get_db():
    # Возвращает базу подсказок для формы
    names = []
    surnames = []
    patronymics = []
    fios = read_runtime_settings().get('names_database', "")
    for line in fios.split('\n'):
        line = line.strip().split(';', maxsplit=1)
        surnames.append(line[0].strip())
        if len(line) > 1:
            line = line[1].strip().split(';', maxsplit=1)
            names.append(line[0].strip())
            if len(line) > 1:
                patronymics.append(line[1].strip())

    names = list(set(names))
    surnames = list(set(surnames))
    patronymics = list(set(patronymics))
    return jsonify(dict(names=names, surnames=surnames, patronymics=patronymics))


from export import mod_export as export_module  # noqa
app.register_blueprint(export_module)

#@app.route('/token')
@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

@app.route('/userinfo')
@fresh_login_required
def get_auth_info():
    me = google.get('userinfo')
    resp = google.authorized_response()
    return jsonify({"data": me.data, "session": str(session), "resp": resp, "is_fresh": session['_fresh']})


if __name__ == "__main__":
    app.run(host='0.0.0.0', processes=10)
    
