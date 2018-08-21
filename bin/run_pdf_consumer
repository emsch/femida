#!/usr/bin/env python3

import pymongo.errors
import argparse
import os
import cv2
import logging
import persistqueue
import femida_detect

parser = argparse.ArgumentParser()
STATUS_IN_PROGRESS = 'ICR in progress'
STATUS_ERROR = 'ICR errored'
STATUS_COMPLETE = 'ICR complete'
STATUS_IN_QUEUE = 'in queue for ICR'
STATUS_PARTIALLY_COMPLETE = 'ICR partially complete'
ANSWERS = 'answers'
PDFS = 'pdfs'
DB = 'femida'

parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=27017)
parser.add_argument('--log-to', type=str,
                    help='path to logfile')
parser.add_argument('--root', type=str, required=True,
                    help='path to local queue')
parser.add_argument('--debug', action='store_true')


def main(args):
    logger = logging.getLogger('consumer :: pdf')
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s - %(message)s')
    if args.log_to is not None:
        handle = logging.FileHandler(args.log_to)
    else:
        handle = logging.StreamHandler()
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    queue = persistqueue.SQLiteQueue(
        os.path.join(args.root, 'pdf'))
    logger.info(f'Opened PDF Queue {queue.path}')
    pdf_status = persistqueue.PDict(
        os.path.join(args.root, 'pdf.status'), 'pdf'
    )
    logger.info(f'Opened PDF Status Dict {pdf_status.path}')
    out_queue = persistqueue.SQLiteQueue(
        os.path.join(args.root, 'answers'))
    logger.info(f'Opened Answers Queue {out_queue.path}')
    out_path = os.path.join(args.root, 'answers', 'img')
    os.makedirs(out_path, exist_ok=True)
    logger.info(f'Storing intermediate images in {out_path}')
    logger.info(f'Connecting to {args.host}:{args.port} with provided credentials')
    conn = pymongo.MongoClient(
        host=args.host,
        port=args.port,
        username=os.environ.get('MONGO_USER'),
        password=os.environ.get('MONGO_PASSWORD')
    )
    pdfs = conn[DB][PDFS]
    logger.info(f'Started listening')
    while True:
        task = queue.get()
        try:
            logger.info(f"Got new task _id={task['_id']}")
            try:
                # try to update mongo database
                pdfs.update_one(
                    {'_id': task['_id']},
                    {'$set': {'status': STATUS_IN_PROGRESS}}
                )
            except pymongo.errors.PyMongoError as e:
                logger.error(f"Task _id={task['_id']} :: recoverable (restart) :: %s", e)
                queue.put(task)
                break
            logger.debug(f"Task _id={task['_id']} :: pdf_to_images_silent")
            try:
                images = femida_detect.pdf2img.pdf_to_images_silent(
                    task['path'],
                    os.path.join(out_path, f"{task['UUID']}__%03d.png")
                )
            except RuntimeError:
                logger.error(f"Task _id={task['_id']} :: status :: {STATUS_ERROR}")
                try:
                    # try to update mongo database
                    pdfs.update_one(
                        {'_id': task['_id']},
                        {'$set': {'status': STATUS_ERROR}}
                    )
                except pymongo.errors.PyMongoError as e:
                    logger.error(f"Task _id={task['_id']} :: recoverable (restart) :: %s", e)
                    queue.put(task)
                    break
                continue
            pdf_status[task['UUID']] = len(images)
            pdf_status[task['UUID']+'__err'] = 0
            logger.debug(f"Task _id={task['_id']} :: converted to {len(images)} images")
            for i, imagef in enumerate(images):
                task_ = dict(
                    UUID=task['UUID'],
                    _id=task['_id'],
                    i=i,
                    imagef=imagef
                )
                out_queue.put(task_)
            logger.info(f"Task _id={task['_id']} :: {len(images)} images have been put in out queue")
        # other possible errors
        except pymongo.errors.PyMongoError as e:
            logger.error(f"recoverable (restart) :: %s", e)


if __name__ == '__main__':
    main(parser.parse_args())
