#!/usr/bin/env python3
import os
import uuid
import datetime
import time
import json
import xlsxwriter
from io import BytesIO

from flask import (
    Flask, request,
    send_from_directory,
    render_template,
    redirect, url_for,
    flash, Response,
    Blueprint, jsonify,
    stream_with_context,
    send_file
)
from flask_login import (
    LoginManager, UserMixin,
    login_required, login_user,
    logout_user,
    current_user
)
from flask_pymongo import PyMongo

from bson.objectid import ObjectId
from bson import json_util
from collections import Counter


mod_export = Blueprint('export', __name__)
from database import mongo  # noqa
pdfs = mongo.db.pdfs
answers = mongo.db.answers


class Col:
    def __init__(self, start_from=0):
        self.i = start_from

    def __call__(self):
        val = self.i
        self.i += 1
        return val

    def current(self):
        return self.i


class Question:
    def __init__(
            self, id_=None, option=None, banned_options="",
            yellow=None, red=None,
    ):
        self.id = id_
        self.banned_options = banned_options
        self.options = Counter([self.clean_option(option)])
        self.has_updates = False
        self.has_contradicting_updates = False
        self.updates = []
        self.yellow = yellow
        self.red = red

    def clean_option(self, option):
        for i in self.banned_options:
            option = option.replace(i, "")
        return option

    def update(self, option):
        if len(self.updates) > 0 and option not in self.updates:
            self.has_contradicting_updates = True
        self.updates.append(option)
        self.options.update([self.clean_option(option)])
        self.has_updates = True

    def get_res_style(self):
        res = self.get_res()
        # ? counts = [i[1] for i in self.options.most_common()]
        if not self.has_contradicting_updates:
            # All good!
            if self.has_updates:
                return res, self.yellow
            else:
                return res, None
        else:
            # PROBLEM!
            return res, self.red

    def get_res(self):
        if len(self.updates) > 0:
            return self.updates[-1]
        else:
            return self.options.most_common(1)[0][0]


@mod_export.route('/export')
def export():
    output = BytesIO()
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()

    red = workbook.add_format({'bold': True, 'bg_color': 'red'})
    yellow = workbook.add_format({'bold': False, 'bg_color': 'yellow'})
    bold = workbook.add_format({'bold': True})

    header = ["№", "status", "surname", "name", "patronumic", "class", "type",
              "variant", "requested_manual", "manual_checks", "img_test_form",
              "img_fio", "UUID"]
    header.extend([str(i) for i in range(1, 41)])
    header.extend(["raw_json"])
    for i, v in enumerate(header):
        worksheet.write(0, i, v, bold)

    for row, r in enumerate(answers.find()):
        col = Col()
        try:
            worksheet.write(1+row, col(), 1+row)
            worksheet.write(1+row, col(), r.get('status', ""))
            # ФИО
            personal = r['personal'][0] if len(r['personal']) > 0 else {}
            worksheet.write(1+row, col(), personal.get('surname', ""))
            worksheet.write(1+row, col(), personal.get('name', ""))
            worksheet.write(1+row, col(), personal.get('patronymic', ""))
            worksheet.write(1+row, col(), personal.get('class', ""))
            worksheet.write(1+row, col(), personal.get('type', ""))
            worksheet.write(1+row, col(), personal.get('variant', ""))
            # cnt
            requested_manual = len(r.get('requested_manual', []))
            if requested_manual > 0:
                worksheet.write(1+row, col(), requested_manual, red)
            else:
                worksheet.write(1+row, col(), requested_manual)
            manual_checks = len(r.get('manual_checks', []))
            worksheet.write(1+row, col(), manual_checks)        
            # IMGS
            worksheet.write(1+row, col(), 'http://femida.emsch.ru' + r.get('img_test_form', ""))
            worksheet.write(1+row, col(), 'http://femida.emsch.ru' + r.get('img_fio', ""))
            worksheet.write(1+row, col(), r.get('UUID', ""))
            # answers
            start = col.current()
            for q in range(1,41):
                question = Question(
                    q, r.get('test_results', {}).get(str(q), ""), "F",
                    yellow, red
                )
                for update in r.get('test_updates', []):
                    if str(q) in update['updates']:
                        question.update(update['updates'][str(q)])
                worksheet.write(1+row, start+q-1, *question.get_res_style())
                col()
            worksheet.write(1+row, col(), json.dumps(r, default=json_util.default))    

        except:
            worksheet.write(1+row, 0, 'ERROR OCCURED.')
            raise

    workbook.close()
    output.seek(0)

    # finally return the file
    attachment_filename='femida_%s.xlsx' % datetime.datetime.now().isoformat()[:19]
    return send_file(output, attachment_filename=attachment_filename, as_attachment=True)
