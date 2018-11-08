#!/usr/bin/env python3
import os
import uuid
import datetime
import time
import json
import random
import errno

from flask import (
    Flask, request,
    send_from_directory,
    render_template,
    redirect, url_for,
    flash, Response,
    jsonify,
    stream_with_context
)
from flask_login import (
    LoginManager, UserMixin,
    login_required, login_user,
    logout_user,
    current_user
)

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
app.secret_key = os.environ['FEMIDA_SECRET_KEY']
app.debug = os.environ.get('FEMIDA_DEBUG', False)

from database import mongo  # noqa
mongo.init_app(app)
pdfs = mongo.db.pdfs
answers = mongo.db.answers

# flask-login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


NAMES_DATABASE_PATH = 'databases/names.csv'


# silly user model
class User(UserMixin):

    def __init__(self, id):
        self.id = id
        self.name = "user" + str(id)
        self.password = self.name + "_secret"

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
            'hand_checks_gap': os.environ.get('FEMIDA_HAND_CHECKS_GAP', 10)
        })
        settings = read_runtime_settings()
    return settings


def update_runtime_settings(**kwargs):
    id_ = read_runtime_settings()['_id']
    mongo.db.runtime_settings.update_one(
        {'_id': id_},
        {'$push': kwargs},
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


@app.route('/form.html')
@login_required
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
    K = read_runtime_settings()['hand_checks_gap']
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
@login_required
def serve_pdf():
    return render_template('pdf.html')


def valid_form(form):
    class_ = form['class']
    variant = form['variant']
    if class_ and not class_.isdigit():
        flash(u'''ОШИБКА: некорректный класс. Должно быть число''')
        return False
    if variant and not variant.isdigit():
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
        "type": form['type'],
        'session_id': session_id, "date": date,
    }
    if form['status'] == 'manual':
        requested_manual = {'session_id': session_id, "date": date}
    else:
        requested_manual = None
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


# somewhere to login
@app.route("/login", methods=["GET", "POST"])
def login():
    # FROM https://github.com/shekhargulati/flask-login-example/blob/master/flask-login-example.py
    # FOR NOW LOGINS ALL USES
    user = User(str(uuid.uuid1())[:8])
    login_user(user, remember=True)
    return redirect(request.args.get("next"))

    # if request.method == 'POST':
    #     username = request.form['username']
    #     password = request.form['password']
    #     if password == username + "_secret":
    #         id = username.split('user')[1]
    #         user = User(id)
    #         login_user(user)
    #         return redirect(request.args.get("next"))
    #     else:
    #         return abort(401)
    # else:
    #     return Response('''
    #     <form action="" method="post">
    #         <p><input type=text name=username>
    #         <p><input type=password name=password>
    #         <p><input type=submit value=Login>
    #     </form>
    #     ''')


# somewhere to logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return Response('<p>Logged out</p>')


# handle login failed
@app.errorhandler(401)
def page_not_found(e):
    return Response('<p>Login failed</p>')


# callback to reload the user object
@login_manager.user_loader
def load_user(userid):
    return User(userid)


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
@login_required
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
@login_required
def serve_monitor():
    if request.method == 'POST':
        form = dict(request.form.items())
        print(form)

        os.makedirs(os.path.dirname(NAMES_DATABASE_PATH), exist_ok=True)
        with open(NAMES_DATABASE_PATH, 'w') as f:
            f.write(form.get('fio_field', ''))
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
        print(data)
        # other column settings -> http://bootstrap-table.wenzhixin.net.cn/documentation/#column-options
        try:
            fios = open(NAMES_DATABASE_PATH).read()
        except IOError:
            fios = ''

        return render_template('monitor.html', table_data=data, table_columns=COLUMNS, fios=fios)


@app.route("/get_db.json")
@login_required
def get_db():
    # Возвращает базу подсказок для формы
    names = []
    surnames = []
    patronymics = []
    with open(NAMES_DATABASE_PATH) as f:
        for line in f:
            surname, name, patronymic = line.strip().split(';')
            names.append(name)
            surnames.append(surname)
            patronymics.append(patronymic)
        names = list(set(names))
        surnames = list(set(surnames))
        patronymics = list(set(patronymics))
        return jsonify(dict(names=names, surnames=surnames, patronymics=patronymics))


from export import mod_export as export_module  # noqa
app.register_blueprint(export_module)


if __name__ == "__main__":
    app.run(host='0.0.0.0', processes=10)
