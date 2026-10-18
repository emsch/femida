"""Fill the local stand's MongoDB with synthetic Femida data.

Idempotent: initiates the single-node replica set when nobody has done it yet,
waits for a primary, and inserts synthetic ``pdfs``/``answers`` only while the
``answers`` collection is empty. Placeholder scans are written as SVG where the
OCR workers would put JPEGs, so the review form has something to display.

Every name below is deliberately artificial. Never replace this with a dump
of the production database: it contains personal data of real children.
"""

import datetime
import os
import random
import sys
import time
import uuid

import pymongo
from pymongo.errors import OperationFailure

MONGO_HOST = os.environ.get("MONGO_HOST", "mongodb")
RESULTS_FOLDER = os.environ.get("RESULTS_FOLDER", "/media/ocr_results/")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/media/pdf_uploads/")
REVIEWER = "reviewer-one@femida.local"
OPTIONS = "ABCDEF"

# pdf_comment follows the pattern enforced on /pdf.html:
# <audience>_<class>_<ОТ|МАТ>_<variant>
BATCHES = [
    ("П1_9_ОТ_101", 40, [
        ("Тестов", "Тест", "Тестович"),
        ("Примерова", "Проверка", "Образцовна"),
        ("Макетов", "Макет", "Макетович"),
    ]),
    ("П2_10_МАТ_202", 30, [
        ("Синтетикова", "Выдумка", "Фиктивовна"),
        ("Заглушкин", "Стенд", "Локалович"),
        ("Черновикова", "Проба", "Пробовна"),
    ]),
    ("П3_11_ОТ_303", 40, [
        ("Условнов", "Пример", "Шаблонович"),
        ("Фейкова", "Имитация", "Муляжевна"),
        ("Болванкин", "Эскиз", "Наброскович"),
    ]),
]


def wait_for_primary(client):
    for _ in range(60):
        try:
            if client.admin.command("hello").get("isWritablePrimary"):
                return
        except pymongo.errors.PyMongoError:
            pass
        time.sleep(1)
    sys.exit("seed: mongodb did not become primary within 60s")


def ensure_replica_set(client):
    try:
        client.admin.command("replSetGetStatus")
        return
    except OperationFailure as e:
        if e.code != 94:  # NotYetInitialized
            raise
    config = {"_id": "rs0", "members": [{"_id": 0, "host": "localhost:27017"}]}
    try:
        client.admin.command("replSetInitiate", config)
        print("seed: replica set initiated")
    except OperationFailure as e:
        print(f"seed: replSetInitiate skipped: {e}")


def svg_fio(path, surname, name, patronymic, class_):
    text = f"{surname} {name} {patronymic}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="220" '
            'viewBox="0 0 1000 220">'
            '<rect width="1000" height="220" fill="#fdfbf3" stroke="#999"/>'
            '<text x="30" y="60" font-size="26" fill="#444">Синтетический бланк '
            '(локальный стенд)</text>'
            f'<text x="30" y="130" font-size="44" font-family="serif" '
            f'font-style="italic" fill="#1a1a6e">{text}</text>'
            f'<text x="30" y="190" font-size="30" fill="#444">Класс: {class_}</text>'
            "</svg>"
        )


def svg_test_form(path, test_results, num_questions):
    rows = []
    for q in range(1, 41):
        y = 30 + (q - 1) * 24
        rows.append(f'<text x="20" y="{y + 6}" font-size="16" fill="#333">{q}</text>')
        for i, letter in enumerate(OPTIONS):
            x = 80 + i * 50
            filled = test_results.get(str(q)) == letter
            fill = "#1a1a6e" if filled else "none"
            rows.append(
                f'<circle cx="{x}" cy="{y}" r="9" fill="{fill}" stroke="#555"/>'
                f'<text x="{x - 5}" y="{y + 5}" font-size="12" '
                f'fill="{"#fff" if filled else "#555"}">{letter}</text>'
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="1000" '
            'viewBox="0 0 400 1000">'
            '<rect width="400" height="1000" fill="#fdfbf3" stroke="#999"/>'
            f'<text x="20" y="12" font-size="11" fill="#777">синтетика, '
            f'{num_questions} вопросов</text>'
            + "".join(rows)
            + "</svg>"
        )


def build_answer(rng, pdf_uuid, page, class_, type_, num_questions, now):
    test_results = {str(q): "" for q in range(1, 41)}
    for q in range(1, num_questions + 1):
        test_results[str(q)] = rng.choice(OPTIONS)
    img_fio = os.path.join(RESULTS_FOLDER, f"{pdf_uuid}__{page}_fio.svg")
    img_test_form = os.path.join(RESULTS_FOLDER, f"{pdf_uuid}__{page}_test_form.svg")
    return {
        "page": page,
        "UUID": pdf_uuid,
        "status": "normal",
        "flipped": False,
        "personal": [{"class": class_, "type": type_}],
        "requested_manual": [],
        "manual_checks": [],
        "test_results": test_results,
        "test_updates": [{"session_id": "ocr", "updates": {}, "date": now}],
        "img_fio": img_fio,
        "img_test_form": img_test_form,
    }


def apply_review(answer, fio, class_, type_, variant, now):
    """Pretend a fictional reviewer already checked this work once."""
    surname, name, patronymic = fio
    answer["personal"].append({
        "class": class_, "name": name, "surname": surname, "patronymic": patronymic,
        "variant": variant, "type": type_, "session_id": REVIEWER, "date": now,
    })
    answer["manual_checks"].append(REVIEWER)
    answer["test_updates"].append({
        "updates": {"1": "F", "2": "NA"}, "date": now, "session_id": REVIEWER,
    })


def seed(db):
    rng = random.Random(2026)
    now = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    pdf_docs = []
    answer_docs = []
    for pdf_comment, num_questions, fios in BATCHES:
        _, class_, type_raw, variant = pdf_comment.split("_")
        type_ = "ot" if type_raw == "ОТ" else "mat"
        pdf_uuid = str(uuid.uuid4())
        pdf_docs.append({
            "UUID": pdf_uuid,
            "path": os.path.join(UPLOAD_FOLDER, f"{pdf_uuid}__{pdf_comment}.pdf"),
            "status": "OCR complete",
            "pdf_comment": pdf_comment,
            "date": now,
            "session_id": REVIEWER,
        })
        for page, fio in enumerate(fios, start=1):
            answer = build_answer(rng, pdf_uuid, page, class_, type_, num_questions, now)
            svg_fio(answer["img_fio"], *fio, class_)
            svg_test_form(answer["img_test_form"], answer["test_results"], num_questions)
            if page == 2:
                apply_review(answer, fio, class_, type_, variant, now)
            if page == 3 and pdf_comment.startswith("П3"):
                answer["requested_manual"].append({
                    "session_id": REVIEWER, "date": now,
                    "comment": "синтетика: бланк залит, нужна ручная проверка",
                })
            answer_docs.append(answer)
    db.pdfs.insert_many(pdf_docs)
    db.answers.insert_many(answer_docs)
    print(f"seed: inserted {len(pdf_docs)} pdfs and {len(answer_docs)} answers")


def main():
    client = pymongo.MongoClient(MONGO_HOST, 27017, directConnection=True,
                                 serverSelectionTimeoutMS=5000)
    ensure_replica_set(client)
    wait_for_primary(client)
    db = client.femida
    if db.answers.count_documents({}) > 0:
        print("seed: answers already present, nothing to do")
        return
    seed(db)


if __name__ == "__main__":
    main()
