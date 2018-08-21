#!/usr/bin/env python3

import pymongo
import pymongo.errors
import argparse
import os
import logging
import persistqueue
parser = argparse.ArgumentParser()

STATUS_WAITING = 'waiting for ICR'
STATUS_IN_QUEUE = 'in queue for ICR'
DB = 'femida'
PDFS = 'pdfs'


parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=27017)
parser.add_argument('--log-to', type=str,
                    help='path to logfile')
parser.add_argument('--root', type=str, required=True,
                    help='path to local queue')
parser.add_argument('--scan-first', action='store_true')
parser.add_argument('--debug', action='store_true')
logger = logging.getLogger('producer')


def main(args):
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
        os.path.join(args.root, 'pdf')
    )
    logger.info(f"Opened PDF Queue in {queue.path}")
    logger.info(f'Connecting to {args.host}:{args.port} with provided credentials')
    conn = pymongo.MongoClient(
        host=args.host,
        port=args.port,
        username=os.environ.get('MONGO_USER'),
        password=os.environ.get('MONGO_PASSWORD')
    )
    pdfs = conn[DB][PDFS]
    try:
        if args.scan_first:
            for item in pdfs.find({'status': STATUS_WAITING}):
                item['skip'] = 0
                logger.info(f"New item for queue _id={item['_id']} "
                            f"on initial scan")
                pdfs.update_one(
                    {'_id': item['_id']},
                    {'$set': {'status': STATUS_IN_QUEUE}}
                )
                queue.put(item)
                logger.debug(f"Item _id={item['_id']} "
                             f"put in queue done")
        with pdfs.watch(
                [{'$match': {'operationType': {'$in': ['insert', 'update', 'replace']}}}]
        ) as stream:
            logger.info(f'Started watching mongo server changes on {DB}::{PDFS}')
            for event in stream:
                from_update = (
                        event['operationType'] == 'update' and
                        'status' in event['updateDescription']['updatedFields'] and
                        event['updateDescription']['updatedFields']['status'] == STATUS_WAITING
                )
                from_insert_or_replace = (
                        event['operationType'] in {'insert', 'replace'} and
                        event['fullDocument']['status'] == STATUS_WAITING
                )
                if from_update or from_insert_or_replace:
                    task = event['fullDocument']
                    task['skip'] = 0
                    logger.info(f"New item for queue _id={event['fullDocument']['_id']} "
                                f"on event {event['operationType']}")
                    pdfs.update_one(
                        event['documentKey'],
                        {'$set': {'status': STATUS_IN_QUEUE}}
                    )
                    queue.put(task)
                    logger.debug(f"Item _id={task['_id']} "
                                 f"put in queue done")
    except pymongo.errors.PyMongoError as e:
        # The ChangeStream encountered an unrecoverable error or the
        # resume attempt failed to recreate the cursor.
        logger.error(f"Task _id={task['_id']} :: recoverable (restart with --scan-first) :: %s", e)


if __name__ == '__main__':
    main(parser.parse_args())
