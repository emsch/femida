import pymongo
import pymongo.errors
import os
import time
import logging
conn = pymongo.MongoClient(
        host=os.environ.get('MONGO_HOST'),
        username=os.environ.get('MONGODB_USERNAME'),
        password=os.environ.get('MONGODB_PASSWORD')
)
logger = logging.getLogger(__name__)


def setup_rs():
    try:
        status = conn.admin.command("replSetGetStatus")
    except pymongo.errors.OperationFailure:
        logger.debug('Failed to configure replica set')
        status = dict(ok=0)
    if status['ok']:
        logger.debug('Replica set is already configured')
        return True
    else:
        try:
            config = {'_id': 'rs0', 'members': [{'_id': 0, 'host': 'localhost:27017'}]}
            conn.admin.command("replSetInitiate", config)
            logger.info('Initiating replica set with implicit configuration')
            return True
        except pymongo.errors.OperationFailure:
            logger.debug('Failed to configure replica set')
            return False


if __name__ == '__main__':
    handle = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s - %(message)s')
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    while not setup_rs():
        time.sleep(10)

