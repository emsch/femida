import logging
import time


def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def tupleit(t):
    return list(map(tupleit, t)) if isinstance(t, (tuple, list)) else t


def _setup_rs(conn, logger):
    import pymongo.errors
    try:
        status = conn.admin.command("replSetGetStatus")
    except pymongo.errors.OperationFailure:
        logger.debug('Failed to configure replica set')
        status = dict(ok=0)
    if status['ok']:
        logger.info('Replica set is already configured')
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


def setup_rs(conn, logger=None, blocking=True):
    logger = logger or logging.root
    logger.info(f'Configuring Replica Set, blocking={blocking}')
    while not _setup_rs(conn, logger):
        if blocking:
            time.sleep(10)
        else:
            return False
    return True
