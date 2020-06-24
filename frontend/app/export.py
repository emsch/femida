#!/usr/bin/env python3
import datetime
import json
import xlsxwriter
from io import BytesIO
import re

from flask import (
    Blueprint,
    send_file
)

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
            yellow=None, red=None, most_common_answer=''
    ):
        self.id = id_
        self.banned_options = banned_options
        if option is None:
            self.options = Counter()
        else:
            self.options = Counter([self.clean_option(option)])
        self.has_updates = False
        self.has_contradicting_updates = False
        self.updates = []
        self.most_common_answer = most_common_answer
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
        if len(self.options) > 1:
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
        if len(self.options) == 0:
            return None
        elif len(self.updates) > 0:
            return self.most_common_answer
        else:
            return self.options.most_common(1)[0][0]

def lower_without_nonalpha(lines):
    reg = re.compile('[^a-zа-я]')
    print([reg.sub('', line.lower()) for line in lines])
    return [reg.sub('', line.lower()) for line in lines]

def fill_sheet(worksheet, row, col, nums_quests, person, type_):
    for field in ['status', 'variant', 'requested_manual', 'manual_checks', 'img_test_form', 'img_fio', 'UUID']:
        worksheet.write(row, col(), *person[type_][field])
    for i in range(1, nums_quests[type_]+1):
        worksheet.write(row, col(), *person[type_][str(i)])
    worksheet.write(row, col(), *person[type_]['raw_json'])
    return worksheet

@mod_export.route('/export')
def export():
    output = BytesIO()
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    
    red = workbook.add_format({'bold': True, 'bg_color': 'red'})
    yellow = workbook.add_format({'bold': False, 'bg_color': 'yellow'})
    bold = workbook.add_format({'bold': True})
    
    header = ["№", "surname", "name", "patronymic", "class"]
    header_type = ["status", "variant", "requested_manual", "manual_checks", "img_test_form",
                   "img_fio", "UUID"]
    nums_quests = {'mat': 30, 'ot': 40}
    type_start_col = {}
    type_start_col['mat'] = len(header)
    type_start_col['ot'] = type_start_col['mat'] + len(header_type) + nums_quests['mat'] + 1 # +1 by "raw_json"
    
    for type_ in ['mat', 'ot']:
        header.extend([f'{field}_{type_}' for field in header_type])
        header.extend([str(i) for i in range(1, nums_quests[type_] + 1)])
        header.append(f'raw_json_{type_}')

    for i, v in enumerate(header):
        worksheet.write(0, i, v, bold)

    all_people = {}
    row = 1
    for r in answers.find():
        col = Col()
        answers_by_checks = r['answers_by_checks']
        personals = r['personal'][1:] # first element is generated by default
        try:
            spec_id = "{}_{}_{}_%s" % answers_by_checks['class'][-1]
            spec_id = spec_id.format(*lower_without_nonalpha([answers_by_checks['name'][-1], answers_by_checks['surname'][-1], answers_by_checks['patronymic'][-1]]))
            print(spec_id)
            type_ = answers_by_checks['type'][-1]
            if  '___' in spec_id:
                # fill document by anonymous users
                worksheet.write(row, col(), row)
                # ФИО
                for field in ['surname', 'name', 'patronymic', 'class']:
                    worksheet.write(row, col(), answers_by_checks[field][-1])
                col.i = type_start_col[type_]
                worksheet.write(row, col(), r.get('status', ""))
                worksheet.write(row, col(), answers_by_checks['variant'][-1])
                # cnt
                requested_manual = len(r.get('requested_manual', 0))
                if requested_manual > 0:
                    worksheet.write(row, col(), requested_manual, red)
                else:
                    worksheet.write(row, col(), requested_manual)
                manual_checks = len(r.get('manual_checks', 0))
                worksheet.write(row, col(), manual_checks)
                # IMGS
                worksheet.write(row, col(), 'http://femida.emsch.ru' + r.get('img_test_form', ""))
                worksheet.write(row, col(), 'http://femida.emsch.ru' + r.get('img_fio', ""))
                worksheet.write(row, col(), r.get('UUID', ""))
                # answers
                start = col.current()
                for q in range(1, nums_quests[type_]+1):
                    question = Question(
                        q, r.get('test_results', {}).get(str(q), ""), "F",
                        yellow, red
                    )
                    for update in r.get('test_updates', []):
                        if str(q) in update['updates']:
                            question.update(update['updates'][str(q)])
                    worksheet.write(row, start+q-1, *question.get_res_style())
                    col()
                worksheet.write(row, col(), json.dumps(r, default=json_util.default))
                row += 1
            else:
                # fill all_people
                if spec_id not in all_people:
                    all_people[spec_id] = {'main': {}, 'mat': {}, 'ot': {}}
                    for field in ['surname', 'name', 'patronymic', 'class', 'variant']:
                        question = Question(field, most_common_answer=answers_by_checks[field][-1], yellow=yellow, red=red)
                        for personal in personals:
                            question.update(personal.get(field, ""))
                        all_people[spec_id]['main' if field != 'variant' else type_][field] = list(question.get_res_style())
                else:
                    for field in ['surname', 'name', 'patronymic', 'class', 'variant']:
                        question = Question(field, most_common_answer=answers_by_checks[field][-1], yellow=yellow, red=red)
                        for personal in personals:
                            question.update(personal.get(field, ""))
                        res_style = question.get_res_style()
                        if field != 'variant':
                            if all_people[spec_id]['main'][field][1] == red or res_style[1] == red:
                                all_people[spec_id]['main'][field][1] = red
                            elif all_people[spec_id]['main'][field][1] == yellow or res_style[1] == yellow:
                                all_people[spec_id]['main'][field][1] = yellow
                        else:
                            all_people[spec_id][type_][field] = res_style
                # cnt
                all_people[spec_id][type_]['status'] = r.get('status', ''), None
                requested_manual = len(r.get('requested_manual', 0))
                if  requested_manual > 0:
                    all_people[spec_id][type_]['requested_manual'] = requested_manual, red
                else:
                    all_people[spec_id][type_]['requested_manual'] = requested_manual, None
                all_people[spec_id][type_]['manual_checks'] = len(r.get('manual_checks', 0)), None
                # IMGS
                all_people[spec_id][type_]['img_test_form'] = 'http://femida.emsch.ru' + r.get('img_test_form', ""), None
                all_people[spec_id][type_]['img_fio'] = 'http://femida.emsch.ru' + r.get('img_fio', ""), None
                all_people[spec_id][type_]['UUID'] = r.get('UUID', ""), None
                # answers
                start = col.current()
                for q in range(1, nums_quests[type_]+1):
                    question = Question(
                        q, r.get('test_results', {}).get(str(q), ""), "F",
                        yellow, red
                    )
                    for update in r.get('test_updates', []):
                        if str(q) in update['updates']:
                            question.update(update['updates'][str(q)])
                    all_people[spec_id][type_][str(q)] = question.get_res_style()
                    col()
                all_people[spec_id][type_]['raw_json'] = json.dumps(r, default=json_util.default), None
        except Exception as e:
            worksheet.write(1+row, 0, 'ERROR OCCURED: ' + str(e))
    
    # fill document by info from all_people
    for person in all_people.values():
        col = Col()
        worksheet.write(row, col(), row)
        for field in ['surname', 'name', 'patronymic', 'class']:
            worksheet.write(row, col(), *person['main'][field])
        if person['mat']:
            worksheet = fill_sheet(worksheet, row, col, nums_quests, person, 'mat')
        if person['ot']:
            col.i = type_start_col['ot']
            worksheet = fill_sheet(worksheet, row, col, nums_quests, person, 'ot')
        row += 1

    workbook.close()
    output.seek(0)
    # finally return the file
    attachment_filename = 'femida_%s.xlsx' % datetime.datetime.now().isoformat()[:19]
    return send_file(output, attachment_filename=attachment_filename, as_attachment=True)
